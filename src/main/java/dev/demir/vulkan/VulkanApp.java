package dev.demir.vulkan;

import dev.demir.vulkan.engine.EngineStats;
import dev.demir.vulkan.engine.VulkanEngine;
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.scene.Animation;
import dev.demir.vulkan.scene.CompactObject;
import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.scene.Geometry;
import dev.demir.vulkan.scene.Material;
import dev.demir.vulkan.scene.MeshLibrary;
import dev.demir.vulkan.scene.ModelInstance;
import dev.demir.vulkan.scene.RenderSettings;
import dev.demir.vulkan.scene.SceneDescription;
import dev.demir.vulkan.scene.ScenePreset;
import dev.demir.vulkan.scene.ScenePresets;
import dev.demir.vulkan.ui.CameraController;
import dev.demir.vulkan.ui.ObjectListPanel;
import dev.demir.vulkan.ui.ObjectPoolPanel;
import dev.demir.vulkan.ui.ObjectSettingsPanel;
import dev.demir.vulkan.ui.SceneSettingsPanel;
import dev.demir.vulkan.ui.SideColumn;
import dev.demir.vulkan.util.Vec3;
import org.lwjgl.system.Configuration;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Application entry point and UI-thread controller.
 *
 * Layout: the rendered image on the left; on the right the Scene box (presets and render
 * settings), the Object pool, the Objects list and the Object settings.
 *
 * Threads: this class runs on the Swing UI thread. Model files load on a background thread
 * (the Scene Rebuild Thread, via CompletableFuture), and all Vulkan work happens on the
 * VulkanEngine's own render thread; they only exchange immutable snapshots.
 */
public class VulkanApp {

    private static final int RENDER_WIDTH = VulkanEngine.WIDTH;
    private static final int RENDER_HEIGHT = VulkanEngine.HEIGHT;

    // --- Engine and scene state (UI thread) ---
    private final AtomicReference<FrameData> latestFrame = new AtomicReference<>();
    private final VulkanEngine engine;
    private final MeshLibrary meshLibrary = new MeshLibrary();
    private final DefaultListModel<ModelInstance> objects = new DefaultListModel<>();
    private final Camera camera = new Camera(new Vec3(0, 10, 40), new Vec3(0, 0, 0), new Vec3(0, 1, 0), 40,
            (double) RENDER_WIDTH / RENDER_HEIGHT);
    private RenderSettings renderSettings = RenderSettings.DEFAULT;
    private CompactObject currentCompactObject; // from the loaded preset, or null

    // --- UI ---
    private JFrame frame;
    private JLabel viewport;
    private int[] imagePixels; // backing array of the displayed image (0x00RRGGBB)
    private SceneSettingsPanel sceneSettingsPanel;
    private ObjectListPanel objectListPanel;
    private ObjectSettingsPanel objectSettingsPanel;

    // --- Rebuild and statistics state (UI thread) ---
    private boolean sceneBuildInProgress = false;
    private boolean rebuildRequested = false; // an edit arrived during a rebuild
    private boolean renderEngineStopped = false;
    private long lastTitleUpdate = System.nanoTime();
    private int displayedFrames = 0;

    public static void main(String[] args) {
        // LWJGL's per-thread MemoryStack defaults to 64 KB. Creating a VkInstance enumerates every
        // device extension onto that stack, and modern drivers (e.g. NVIDIA, Vulkan 1.4) list
        // enough of them to overflow it. Must be set before any LWJGL class uses the stack.
        Configuration.STACK_SIZE.set(512); // in KB

        try {
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to set Nimbus L&F: " + e.getMessage());
        }

        SwingUtilities.invokeLater(() -> {
            try {
                new VulkanApp().run();
            } catch (Exception e) {
                e.printStackTrace();
                JOptionPane.showMessageDialog(null, "Critical Error: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                System.exit(1);
            }
        });
    }

    public VulkanApp() {
        this.engine = new VulkanEngine(latestFrame,
                error -> SwingUtilities.invokeLater(() -> showRenderEngineError(error)));
    }

    public void run() {
        frame = new JFrame("Vulkan Ray Tracer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        BufferedImage image = new BufferedImage(RENDER_WIDTH, RENDER_HEIGHT, BufferedImage.TYPE_INT_RGB);
        imagePixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        viewport = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(viewport, BorderLayout.CENTER);
        frame.getContentPane().add(createSidePanel(), BorderLayout.EAST);
        CameraController.attach(viewport, camera, () -> engine.submitCamera(camera.state()));

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        engine.start();
        loadPreset(ScenePresets.custom());

        Timer uiTimer = new Timer(16, e -> updateUI());
        uiTimer.start();
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                uiTimer.stop();
                engine.stop();
            }
        });
    }

    private JComponent createSidePanel() {
        sceneSettingsPanel = new SceneSettingsPanel(ScenePresets.all(), this::loadPreset, this::updateSettings);
        ObjectPoolPanel poolPanel = new ObjectPoolPanel(camera::getLookAt, this::addObject);
        objectListPanel = new ObjectListPanel(objects,
                selected -> objectSettingsPanel.show(selected), this::duplicateObject, this::removeObject);
        objectSettingsPanel = new ObjectSettingsPanel(this::objectsChanged, this::objectNames);

        SideColumn column = new SideColumn(sceneSettingsPanel, poolPanel, objectListPanel, objectSettingsPanel);
        JScrollPane scroll = new JScrollPane(column, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        scroll.setPreferredSize(new Dimension(400, RENDER_HEIGHT));
        return scroll;
    }

    // --- Scene editing (UI thread) ---

    private void loadPreset(ScenePreset preset) {
        objects.clear();
        for (ModelInstance instance : preset.objects().get()) {
            objects.addElement(instance);
        }
        objectSettingsPanel.show(null);
        camera.setOrigin(preset.cameraOrigin());
        camera.setLookAt(preset.cameraLookAt());
        camera.setVfov(preset.vfov());
        currentCompactObject = preset.compactObject();
        engine.submitCamera(camera.state());
        updateSettings(preset.settings());
        sceneSettingsPanel.show(preset.settings(), preset);
        rebuildSceneAsync();
    }

    private void addObject(ModelInstance instance) {
        instance.setDisplayName(uniqueName(instance.getDisplayName()));
        objects.addElement(instance);
        objectListPanel.select(instance);
        rebuildSceneAsync();
    }

    private void duplicateObject(ModelInstance original) {
        ModelInstance copy = original.copy(original.getDisplayName());
        Vec3 p = original.getPosition();
        double offset = 2.5 * Math.max(original.getScale().x, Math.max(original.getScale().y, original.getScale().z));
        copy.setPosition(new Vec3(p.x + offset, p.y, p.z));
        addObject(copy);
    }

    private void removeObject(ModelInstance instance) {
        objects.removeElement(instance);
        objectSettingsPanel.show(null);
        rebuildSceneAsync();
    }

    private void objectsChanged() {
        objectListPanel.refresh();
        rebuildSceneAsync();
    }

    /** "Human" -> "Human 2" -> "Human 3" ... (also when duplicating "Human 2"). */
    private String uniqueName(String requested) {
        String base = requested.replaceFirst(" \\d+$", "");
        String name = requested;
        for (int n = 2; contains(name); n++) {
            name = base + " " + n;
        }
        return name;
    }

    private List<String> objectNames() {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < objects.size(); i++) {
            names.add(objects.get(i).getDisplayName());
        }
        return names;
    }

    private boolean contains(String name) {
        for (int i = 0; i < objects.size(); i++) {
            if (objects.get(i).getDisplayName().equals(name)) return true;
        }
        return false;
    }

    /** Sends new render settings to the engine (which restarts accumulation). */
    private void updateSettings(RenderSettings settings) {
        renderSettings = settings;
        engine.submitSettings(settings);
    }

    // --- Scene rebuild (UI thread -> rebuild thread -> render thread) ---

    /** Values captured on the UI thread, so the rebuild thread never reads objects being edited. */
    private record InstanceSpec(String modelPath, String name, Vec3 position, Vec3 rotation, Vec3 scale,
                                Material material, Animation animation) {
    }

    /**
     * Rebuilds the scene in the background: loads any new model files (cached by MeshLibrary)
     * and hands an immutable SceneDescription to the render thread, which then only builds
     * BLASes for new geometry and rebuilds the TLAS. A request that arrives while a rebuild
     * runs is remembered and run right after, so no edit is lost.
     */
    private void rebuildSceneAsync() {
        if (sceneBuildInProgress) {
            rebuildRequested = true;
            return;
        }
        sceneBuildInProgress = true;
        rebuildRequested = false;

        CompactObject compactObject = currentCompactObject; // captured on the UI thread
        List<InstanceSpec> specs = new ArrayList<>();
        for (int i = 0; i < objects.size(); i++) {
            ModelInstance instance = objects.get(i);
            specs.add(new InstanceSpec(instance.getModelPath(), instance.getDisplayName(), instance.getPosition(),
                    instance.getRotation(), instance.getScale(), instance.getMaterial(), instance.getAnimation()));
        }

        CompletableFuture
                .supplyAsync(() -> loadSceneDescription(specs, compactObject))
                .whenCompleteAsync((description, error) -> {
                    if (error != null) {
                        error.printStackTrace();
                        JOptionPane.showMessageDialog(frame, "Scene build failed: " + error.getMessage(),
                                "Scene Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        engine.submitScene(description); // the engine restarts accumulation
                    }
                    sceneBuildInProgress = false;
                    if (rebuildRequested) {
                        rebuildSceneAsync();
                    }
                }, SwingUtilities::invokeLater);
    }

    /**
     * Runs on the rebuild thread: resolves geometry (loading new files), skips models that fail
     * to load, and turns orbit center names into instance indices.
     */
    private SceneDescription loadSceneDescription(List<InstanceSpec> specs, CompactObject compactObject) {
        List<InstanceSpec> loaded = new ArrayList<>();
        List<Geometry> geometries = new ArrayList<>();
        for (InstanceSpec spec : specs) {
            try {
                geometries.add(meshLibrary.get(spec.modelPath()));
                loaded.add(spec);
            } catch (RuntimeException e) {
                System.err.println("WARN (SRT): Skipping '" + spec.name() + "': " + e.getMessage());
            }
        }
        Map<String, Integer> indexByName = new HashMap<>();
        for (int i = 0; i < loaded.size(); i++) {
            indexByName.putIfAbsent(loaded.get(i).name(), i);
        }

        List<SceneDescription.Instance> instances = new ArrayList<>();
        for (int i = 0; i < loaded.size(); i++) {
            InstanceSpec spec = loaded.get(i);
            Animation animation = spec.animation();
            int parent = SceneDescription.NO_ORBIT;
            if (animation.orbits()) {
                if (Animation.WORLD_ORIGIN.equals(animation.orbitCenter())) {
                    parent = SceneDescription.ORBIT_WORLD_ORIGIN;
                } else {
                    Integer index = indexByName.get(animation.orbitCenter());
                    if (index == null || index == i) {
                        System.err.println("WARN (SRT): '" + spec.name() + "' orbits unknown object '"
                                + animation.orbitCenter() + "'; ignoring the orbit.");
                    } else {
                        parent = index;
                    }
                }
            }
            instances.add(new SceneDescription.Instance(geometries.get(i), spec.position(), spec.rotation(),
                    spec.scale(), spec.material(), parent, animation.orbitPeriod(), animation.spinPeriod()));
        }
        return new SceneDescription(instances, compactObject);
    }

    // --- Display ---

    /**
     * UI loop (Swing timer, ~60 Hz): shows the newest finished frame and, once per second,
     * the statistics. The render thread runs independently and may render many more frames.
     */
    private void updateUI() {
        FrameData frameData = latestFrame.getAndSet(null);
        if (frameData != null) {
            System.arraycopy(frameData.pixels(), 0, imagePixels, 0, imagePixels.length);
            engine.recycleFrame(frameData);
            viewport.repaint();
            displayedFrames++;
        }
        long now = System.nanoTime();
        if (!renderEngineStopped && now - lastTitleUpdate >= 1_000_000_000L) {
            double displayFps = displayedFrames * 1e9 / (now - lastTitleUpdate);
            EngineStats stats = engine.stats();
            boolean converged = renderSettings.maxSamples() > 0 && stats.samples() >= renderSettings.maxSamples();
            frame.setTitle(String.format("Vulkan Ray Tracer | Render %.0f FPS | GPU %.2f ms | Display %.0f FPS | Samples %d%s",
                    stats.renderFps(), stats.gpuMs(), displayFps, stats.samples(), converged ? " (done)" : ""));
            displayedFrames = 0;
            lastTitleUpdate = now;
        }
    }

    /** Called on the UI thread when the render thread has crashed; without this the window would just stay black. */
    private void showRenderEngineError(Throwable error) {
        renderEngineStopped = true;
        if (frame != null) {
            frame.setTitle("Vulkan Ray Tracer | Render engine stopped");
        }
        JOptionPane.showMessageDialog(frame,
                "The Vulkan render engine stopped:\n" + error + "\n\nSee the console for the full stack trace.",
                "Render Engine Error", JOptionPane.ERROR_MESSAGE);
    }
}
