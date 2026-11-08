package dev.demir.vulkan;

import dev.demir.vulkan.engine.EngineStats;
import dev.demir.vulkan.engine.RenderSettings;
import dev.demir.vulkan.engine.VulkanEngine;
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.scene.Geometry;
import dev.demir.vulkan.scene.Material;
import dev.demir.vulkan.scene.MeshLibrary;
import dev.demir.vulkan.scene.SphereGeometry;
import dev.demir.vulkan.scene.ModelInstance;
import dev.demir.vulkan.scene.Scene;
import dev.demir.vulkan.scene.SceneDescription;
import dev.demir.vulkan.util.Vec3;
import org.lwjgl.system.Configuration;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Phase 5: (UI Refactor)
 * 1. (REQUEST) Re-organized control panel into "Global Settings" and
 * "Object Properties" using GridBagLayout for a cleaner look.
 * 2. (REQUEST) Added a stubbed "Exposure" JSlider to Global Settings.
 * 3. All other logic (Sky Toggle, Accumulation, Keys) is retained.
 *
 * (3-THREAD-RACE-CONDITION-FIX): All state management (frameCount, skyEnabled)
 * is now controlled ONLY by this class (the UI Thread).
 * VulkanEngine is now a "dumb" renderer.
 * Accumulation is now paused during Scene Rebuild (SRT).
 */
public class VulkanApp {

    // --- Configuration ---
    private static final int RENDER_WIDTH = 1280;
    private static final int RENDER_HEIGHT = 720;

    // --- Swing UI ---
    private JFrame frame;
    private JLabel imageLabel;
    private BufferedImage bufferedImage;
    private int[] imagePixels; // backing array of bufferedImage (0x00RRGGBB)

    // --- Scene & UI Controls ---
    private JList<ModelInstance> sceneObjectList;
    private DefaultListModel<ModelInstance> listModel;
    private JPanel controlPanel;
    private JSpinner posXSpinner, posYSpinner, posZSpinner;
    private JSpinner uniformScaleSpinner;
    private JComboBox<ColorPreset> colorComboBox;
    private JPanel rgbPanel;
    private JSpinner colorRSpinner, colorGSpinner, colorBSpinner;
    private JComboBox<MaterialPreset> materialComboBox;
    // --- NEW: Global Controls ---
    private JCheckBox enableSkyCheckBox;
    private JSlider exposureSlider;

    private JButton applyChangesButton;
    private ModelInstance selectedInstance = null;
    private boolean isUpdatingUI = false; // Flag to prevent UI event loops

    // --- Engine Core ---
    private final AtomicReference<FrameData> latestFrame = new AtomicReference<>();
    private final VulkanEngine vulkanEngine;
    private final MeshLibrary meshLibrary = new MeshLibrary();
    private final Camera camera;
    private final Scene scene = new Scene();

    // --- State ---
    private RenderSettings renderSettings = RenderSettings.DEFAULT;
    private long lastTitleUpdate = System.nanoTime();
    private int displayedFrames = 0;

    private boolean sceneBuildInProgress = false; // UI thread only
    private boolean rebuildRequested = false; // UI thread only: an edit arrived during a rebuild

    // Set (on the UI thread) when the render thread has crashed; stops the FPS title updates.
    private boolean renderEngineStopped = false;

    // --- Helper class for Color ComboBox ---
    private static class ColorPreset {
        private final String name;
        private final Vec3 color;
        public ColorPreset(String name, Vec3 color) { this.name = name; this.color = color; }
        public String getName() { return name; }
        public Vec3 getColor() { return color; }
        @Override
        public String toString() { return name; } // This shows in the ComboBox
    }

    // --- Helper class for Material ComboBox ---
    private static class MaterialPreset {
        private final String name;
        private final float type;
        public MaterialPreset(String name, float type) { this.name = name; this.type = type; }
        public String getName() { return name; }
        public float getType() { return type; }
        @Override
        public String toString() { return name; }
    }


    public static void main(String[] args) {
        // LWJGL's per-thread MemoryStack defaults to 64 KB. Creating a VkInstance enumerates every
        // device extension onto that stack, and modern drivers (e.g. NVIDIA, Vulkan 1.4) list
        // enough of them to overflow it. Must be set before any LWJGL class uses the stack.
        Configuration.STACK_SIZE.set(512); // in KB

        // Set Nimbus Look and Feel
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
        this.vulkanEngine = new VulkanEngine(latestFrame,
                error -> SwingUtilities.invokeLater(() -> showRenderEngineError(error)));
        this.camera = new Camera(
                new Vec3(-25, 30, 140), // origin
                new Vec3(0, 0, 0),  // lookAt
                new Vec3(0, 1, 0),  // vUp
                20.0,               // vfov
                (double) RENDER_WIDTH / RENDER_HEIGHT
        );
    }

    public void run() {
        // 1. Set up Swing UI
        frame = new JFrame("Vulkan Ray Tracer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        bufferedImage = new BufferedImage(RENDER_WIDTH, RENDER_HEIGHT, BufferedImage.TYPE_INT_RGB);
        imageLabel = new JLabel(new ImageIcon(bufferedImage));
        frame.getContentPane().add(imageLabel, BorderLayout.CENTER);

        // 1b. Set up the Scene Control Panel
        setupSceneControls();
        frame.getContentPane().add(controlPanel, BorderLayout.EAST);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        imagePixels = ((DataBufferInt) bufferedImage.getRaster().getDataBuffer()).getData();

        // 2. Start the render thread and give it camera, settings and the first scene
        vulkanEngine.start();
        vulkanEngine.submitCamera(camera.state());
        vulkanEngine.submitSettings(renderSettings);
        populateDefaultScene();
        rebuildSceneAsync();

        // 5. Start UI Timer
        Timer swingTimer = new Timer(16, (e) -> updateUI());
        swingTimer.start();
        System.out.println("LOG (UI-RUN): UI Timer started.");

        // 6. Set up keyboard controls
        setupKeyBindings();

        // 7. Handle window closing
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                System.out.println("LOG (UI-WINDOW): Stopping engine...");
                swingTimer.stop();
                vulkanEngine.stop();
            }
        });
    }

    /**
     * UI loop (Swing timer, ~60 Hz): shows the newest finished frame and, once per second,
     * the statistics. The render thread runs independently and may render many more frames.
     */
    private void updateUI() {
        FrameData frameData = latestFrame.getAndSet(null);
        if (frameData != null) {
            System.arraycopy(frameData.pixels(), 0, imagePixels, 0, imagePixels.length);
            vulkanEngine.recycleFrame(frameData);
            imageLabel.repaint();
            displayedFrames++;
        }
        long now = System.nanoTime();
        if (!renderEngineStopped && now - lastTitleUpdate >= 1_000_000_000L) {
            double displayFps = displayedFrames * 1e9 / (now - lastTitleUpdate);
            EngineStats stats = vulkanEngine.stats();
            frame.setTitle(String.format("Vulkan Ray Tracer | Render %.0f FPS | GPU %.2f ms | Display %.0f FPS | Samples %d",
                    stats.renderFps(), stats.gpuMs(), displayFps, stats.samples()));
            displayedFrames = 0;
            lastTitleUpdate = now;
        }
    }

    /**
     * Called on the UI thread when the Vulkan render thread has crashed.
     * Without this the window would just stay black.
     */
    private void showRenderEngineError(Throwable error) {
        renderEngineStopped = true;
        if (frame != null) {
            frame.setTitle("Vulkan Ray Tracer | Render engine stopped");
        }
        JOptionPane.showMessageDialog(frame,
                "The Vulkan render engine stopped:\n" + error + "\n\nSee the console for the full stack trace.",
                "Render Engine Error", JOptionPane.ERROR_MESSAGE);
    }

    /** Instance values captured on the UI thread, so the rebuild thread never reads live UI objects. */
    private record InstanceSpec(String modelPath, String name, float[] transform, float[] albedo, float materialType) {
    }

    /**
     * Rebuilds the scene in the background (SRT): loads any new model files (cached by
     * MeshLibrary) and hands an immutable SceneDescription to the render thread, which
     * then only has to build BLASes for new meshes and rebuild the TLAS.
     * A request that arrives while a rebuild runs is remembered and run right after.
     */
    public void rebuildSceneAsync() {
        if (sceneBuildInProgress) {
            rebuildRequested = true;
            return;
        }
        sceneBuildInProgress = true;
        rebuildRequested = false;

        List<InstanceSpec> specs = new ArrayList<>();
        for (ModelInstance instance : scene.getInstances()) {
            Vec3 color = instance.getColor();
            specs.add(new InstanceSpec(instance.getModelPath(), instance.getDisplayName(), instance.toTransform3x4(),
                    new float[]{(float) color.x, (float) color.y, (float) color.z}, instance.getMaterialType()));
        }

        CompletableFuture
                .supplyAsync(() -> loadSceneDescription(specs))
                .whenCompleteAsync((description, error) -> {
                    if (error != null) {
                        error.printStackTrace();
                        JOptionPane.showMessageDialog(frame, "Scene build failed: " + error.getMessage(),
                                "Scene Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        vulkanEngine.submitScene(description); // the engine restarts accumulation
                    }
                    sceneBuildInProgress = false;
                    if (rebuildRequested) {
                        rebuildSceneAsync();
                    }
                }, SwingUtilities::invokeLater);
    }

    /** Runs on the rebuild thread: resolves meshes (loading new files) and skips models that fail to load. */
    private SceneDescription loadSceneDescription(List<InstanceSpec> specs) {
        List<SceneDescription.Instance> instances = new ArrayList<>();
        for (InstanceSpec spec : specs) {
            try {
                Geometry geometry = meshLibrary.get(spec.modelPath());
                instances.add(new SceneDescription.Instance(geometry, spec.transform(),
                        Material.fromLegacy(spec.albedo(), spec.materialType())));
            } catch (RuntimeException e) {
                System.err.println("WARN (SRT): Skipping '" + spec.name() + "': " + e.getMessage());
            }
        }
        return new SceneDescription(instances);
    }

    /**
     * Loads the default scene from models committed to the repo
     * ('ground_plane.obj', 'FinalBaseMesh.obj'). 'car.obj' and 'sun.obj' are
     * gitignored, so they can only be added with "Add Model" when present locally.
     */
    private void populateDefaultScene() {
        System.out.println("LOG (UI-POPULATE): Populating default scene...");

        // --- GROUND (Matte) ---
        ModelInstance plane = new ModelInstance("./objects/ground_plane.obj", "Ground Plane");
        plane.setPosition(new Vec3(0, -10, 0));
        plane.setScale(new Vec3(150, 1, 150)); // Non-uniform scale
        plane.setColor(new Vec3(0.5, 0.5, 0.5)); // "Grey" preset
        plane.setMaterialType(0.0f); // 0.0f = Lambertian (Matte)
        scene.addInstance(plane);
        listModel.addElement(plane);

        // --- HUMAN (Metal) --- (mesh is ~20.7 units tall, feet at y = 0)
        ModelInstance human = new ModelInstance("./objects/FinalBaseMesh.obj", "Human");
        human.setPosition(new Vec3(0, -10, 0));
        human.setScale(new Vec3(1.5, 1.5, 1.5));
        human.setColor(new Vec3(0.6, 0.7, 0.1));
        human.setMaterialType(1.0f); // 1.0f = Metal
        scene.addInstance(human);
        listModel.addElement(human);

        /*
        // --- LIGHT SOURCE (Emissive) ---
        ModelInstance light = new ModelInstance("./objects/sun.obj", "Light Source");
        light.setPosition(new Vec3(0, 220, 0));
        light.setScale(new Vec3(0.35, 0.35, 0.35));
        light.setColor(new Vec3(4.0, 4.0, 4.0)); // Bright white ( > 1.0)
        light.setMaterialType(3.0f); // 3.0f = Emissive
        scene.addInstance(light);
        listModel.addElement(light);
         */


        sceneObjectList.setSelectedValue(human, true); // Select the human by default
    }


    // --- UI HELPER METHODS (REFACTORED FOR GridBagLayout) ---

    /**
     * Creates the main control panel and all its sub-panels.
     * REFACTORED for a cleaner GridBagLayout.
     */
    private void setupSceneControls() {
        controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
        controlPanel.setPreferredSize(new Dimension(300, RENDER_HEIGHT));
        controlPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // --- 1. Scene Object List Panel ---
        JPanel listPanel = new JPanel(new BorderLayout());
        listPanel.setBorder(new TitledBorder("Scene Objects"));
        listModel = new DefaultListModel<>();
        sceneObjectList = new JList<>(listModel);
        sceneObjectList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        listPanel.add(new JScrollPane(sceneObjectList), BorderLayout.CENTER);
        sceneObjectList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                selectedInstance = sceneObjectList.getSelectedValue();
                updateSpinnersFromInstance();
            }
        });

        // --- 2. List Control Buttons ---
        JPanel buttonPanel = new JPanel(new FlowLayout());
        JButton addButton = new JButton("Add Model");
        addButton.addActionListener(e -> addModelInstance());
        JButton removeButton = new JButton("Remove");
        removeButton.addActionListener(e -> removeSelectedInstance());
        JButton addSphereButton = new JButton("Add Sphere");
        addSphereButton.addActionListener(e -> {
            ModelInstance sphere = new ModelInstance(SphereGeometry.KEY, "Sphere");
            sphere.setScale(new Vec3(5, 5, 5));
            sphere.setPosition(new Vec3(15, -5, 0));
            scene.addInstance(sphere);
            listModel.addElement(sphere);
            sceneObjectList.setSelectedValue(sphere, true);
            rebuildSceneAsync();
        });
        buttonPanel.add(addButton);
        buttonPanel.add(addSphereButton);
        buttonPanel.add(removeButton);

        // --- 3. Global Settings Panel (NEW) ---
        JPanel globalSettingsPanel = createGlobalSettingsPanel();

        // --- 4. Object Properties Panel ---
        JPanel propertiesPanel = createObjectPropertiesPanel();

        // --- Add all panels to the main control panel ---
        controlPanel.add(listPanel);
        controlPanel.add(buttonPanel);
        controlPanel.add(globalSettingsPanel); // Add new global panel
        controlPanel.add(propertiesPanel);
        controlPanel.add(Box.createVerticalGlue()); // Push controls to the top
    }

    /**
     * NEW: Creates the "Global Settings" panel.
     * (3-THREAD-FIX): This is the critical fix.
     */
    private JPanel createGlobalSettingsPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(new TitledBorder("Global Settings"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 5, 2, 5);
        gbc.weightx = 1.0;

        // --- Enable Sky Light CheckBox ---
        enableSkyCheckBox = new JCheckBox("Enable Sky Light", true);
        enableSkyCheckBox.addActionListener(e -> updateSettings(renderSettings.withSky(enableSkyCheckBox.isSelected())));
        addLabelAndComponent(panel, "", enableSkyCheckBox, gbc, 0); // No label

        // --- Exposure (Composure) Slider ---
        exposureSlider = new JSlider(JSlider.HORIZONTAL, -50, 50, 0); // Range -5.0 to +5.0 (x10)
        exposureSlider.setMajorTickSpacing(25);
        exposureSlider.setMinorTickSpacing(5);
        exposureSlider.setPaintTicks(true);
        exposureSlider.setPaintLabels(false); // Labels would be -50, 0, 50. Not user-friendly.

        exposureSlider.addChangeListener(e -> {
            if (!exposureSlider.getValueIsAdjusting()) {
                updateSettings(renderSettings.withExposure(exposureSlider.getValue() / 10.0f));
            }
        });
        addLabelAndComponent(panel, "Exposure:", exposureSlider, gbc, 1);

        return panel;
    }

    /**
     * NEW: Creates the "Object Properties" panel.
     */
    private JPanel createObjectPropertiesPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(new TitledBorder("Object Properties"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 5, 2, 5);
        gbc.weightx = 1.0;
        int row = 0;

        // --- Create all components ---
        posXSpinner = new JSpinner(new SpinnerNumberModel(0.0, -1000.0, 1000.0, 1.0));
        posYSpinner = new JSpinner(new SpinnerNumberModel(0.0, -1000.0, 1000.0, 1.0));
        posZSpinner = new JSpinner(new SpinnerNumberModel(0.0, -1000.0, 1000.0, 1.0));
        uniformScaleSpinner = new JSpinner(new SpinnerNumberModel(1.0, -100.0, 100.0, 0.1));

        colorComboBox = new JComboBox<>(new ColorPreset[]{
                new ColorPreset("Grey", new Vec3(0.5, 0.5, 0.5)),
                new ColorPreset("White", new Vec3(1.0, 1.0, 1.0)),
                new ColorPreset("Red", new Vec3(1.0, 0.0, 0.0)),
                new ColorPreset("Green", new Vec3(0.0, 1.0, 0.0)),
                new ColorPreset("Blue", new Vec3(0.0, 0.0, 1.0)),
                new ColorPreset("Custom...", null)
        });

        materialComboBox = new JComboBox<>(new MaterialPreset[]{
                new MaterialPreset("Matte (Lambertian)", 0.0f),
                new MaterialPreset("Metal (Shiny)", 1.0f),
                new MaterialPreset("Metal (Fuzzy)", 2.0f),
                new MaterialPreset("Emissive (Light)", 3.0f),
                new MaterialPreset("Glass", 4.0f)
        });

        // --- Add components to panel using GridBagLayout ---
        addLabelAndComponent(panel, "Pos X:", posXSpinner, gbc, row++);
        addLabelAndComponent(panel, "Pos Y:", posYSpinner, gbc, row++);
        addLabelAndComponent(panel, "Pos Z:", posZSpinner, gbc, row++);
        addLabelAndComponent(panel, "Scale:", uniformScaleSpinner, gbc, row++);
        addLabelAndComponent(panel, "Color:", colorComboBox, gbc, row++);

        // --- Custom RGB Panel ---
        rgbPanel = new JPanel(new GridBagLayout());
        colorRSpinner = new JSpinner(new SpinnerNumberModel(0.8, 0.0, 1.0, 0.01));
        colorGSpinner = new JSpinner(new SpinnerNumberModel(0.8, 0.0, 1.0, 0.01));
        colorBSpinner = new JSpinner(new SpinnerNumberModel(0.8, 0.0, 1.0, 0.01));

        // Add RGB spinners to their own panel
        GridBagConstraints rgbGbc = new GridBagConstraints();
        rgbGbc.insets = new Insets(0, 2, 0, 2);
        rgbGbc.weightx = 1.0;
        addLabelAndComponent(rgbPanel, "R:", colorRSpinner, rgbGbc, 0);
        addLabelAndComponent(rgbPanel, "G:", colorGSpinner, rgbGbc, 1);
        addLabelAndComponent(rgbPanel, "B:", colorBSpinner, rgbGbc, 2);

        // Add the rgbPanel itself to the main properties panel
        gbc.gridx = 1; // Align under the ComboBox
        gbc.gridy = row++;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(rgbPanel, gbc);
        rgbPanel.setVisible(false);

        // Color ComboBox listener
        colorComboBox.addActionListener(e -> {
            ColorPreset selected = (ColorPreset) colorComboBox.getSelectedItem();
            rgbPanel.setVisible(selected != null && selected.getName().equals("Custom..."));
            frame.pack(); // Adjust window size
        });

        addLabelAndComponent(panel, "Material:", materialComboBox, gbc, row++);

        // --- Add "Apply Changes" button ---
        applyChangesButton = new JButton("Apply Changes & Rebuild");
        applyChangesButton.addActionListener(e -> {
            System.out.println("LOG (UI-EVENT-APPLY): 'Apply Changes' clicked.");
            applyChangesToInstance();
            // This is an "event". It triggers the (now safe) rebuild.
            rebuildSceneAsync(); // This will handle the reset
        });

        gbc.gridx = 0;
        gbc.gridy = row++;
        gbc.gridwidth = 2; // Span both columns
        gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(applyChangesButton, gbc);

        // Add listeners to spinners for "Enter" key
        addSpinnerEnterListener(posXSpinner);
        addSpinnerEnterListener(posYSpinner);
        addSpinnerEnterListener(posZSpinner);
        addSpinnerEnterListener(uniformScaleSpinner);
        addSpinnerEnterListener(colorRSpinner);
        addSpinnerEnterListener(colorGSpinner);
        addSpinnerEnterListener(colorBSpinner);

        return panel;
    }

    /**
     * NEW: Helper to add a label and component in a clean grid row.
     */
    private void addLabelAndComponent(Container container, String text, Component component, GridBagConstraints gbc, int row) {
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 1;
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        container.add(new JLabel(text), gbc);

        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.EAST;
        container.add(component, gbc);
    }

    /**
     * NEW: Helper to add the "Enter" key listener to a spinner's text field.
     */
    private void addSpinnerEnterListener(JSpinner spinner) {
        JFormattedTextField ftf = ((JSpinner.NumberEditor) spinner.getEditor()).getTextField();
        ftf.addActionListener(e -> {
            System.out.println("LOG (UI-EVENT-SPINNER): 'Enter' pressed on spinner.");
            applyChangesToInstance();
            // This is an "event".
            rebuildSceneAsync(); // This will handle the reset
        });
    }

    /**
     * Action for the "Add Model" button.
     */
    private void addModelInstance() {
        JFileChooser fileChooser = new JFileChooser(".");
        fileChooser.setDialogTitle("Select an OBJ Model File");
        fileChooser.setFileFilter(new FileFilter() {
            public boolean accept(File f) { return f.isDirectory() || f.getName().toLowerCase().endsWith(".obj"); }
            public String getDescription() { return "OBJ Models (*.obj)"; }
        });
        int result = fileChooser.showOpenDialog(frame);
        if (result == JFileChooser.APPROVE_OPTION) {
            System.out.println("LOG (UI-EVENT-ADD): 'Add Model' approved.");
            File file = fileChooser.getSelectedFile();
            ModelInstance instance = new ModelInstance(file.getPath(), file.getName());
            scene.addInstance(instance);
            listModel.addElement(instance);
            sceneObjectList.setSelectedValue(instance, true);
            rebuildSceneAsync(); // This will handle the reset
        }
    }

    /**
     * Action for the "Remove" button.
     */
    private void removeSelectedInstance() {
        if (selectedInstance != null) {
            System.out.println("LOG (UI-EVENT-REMOVE): Removing model: " + selectedInstance.getDisplayName());
            scene.removeInstance(selectedInstance);
            listModel.removeElement(selectedInstance);
            selectedInstance = null;
            rebuildSceneAsync(); // This will handle the reset
        }
    }


    /**
     * Action for the "Apply Changes" button.
     * Reads values from spinners and saves them to the selected ModelInstance.
     */
    private void applyChangesToInstance() {
        if (selectedInstance == null || isUpdatingUI) return;
        System.out.println("LOG (UI-APPLY): Applying property changes to " + selectedInstance.getDisplayName());

        // Position
        selectedInstance.setPosition(new Vec3(
                (Double) posXSpinner.getValue(),
                (Double) posYSpinner.getValue(),
                (Double) posZSpinner.getValue()
        ));

        // Scale
        double scaleValue = (Double) uniformScaleSpinner.getValue();
        selectedInstance.setScale(new Vec3(scaleValue, scaleValue, scaleValue));

        // Color
        ColorPreset preset = (ColorPreset) colorComboBox.getSelectedItem();
        if (preset != null && preset.getColor() != null) {
            selectedInstance.setColor(preset.getColor());
        } else {
            selectedInstance.setColor(new Vec3(
                    (Double) colorRSpinner.getValue(),
                    (Double) colorGSpinner.getValue(),
                    (Double) colorBSpinner.getValue()
            ));
        }

        // Apply Material Type
        MaterialPreset matPreset = (MaterialPreset) materialComboBox.getSelectedItem();
        if (matPreset != null) {
            selectedInstance.setMaterialType(matPreset.getType());
        }
    }

    /**
     * Updates the spinner controls to reflect the selected object's properties.
     */
    private void updateSpinnersFromInstance() {
        if (selectedInstance == null) return;
        isUpdatingUI = true; // Set flag to prevent feedback loop
        // System.out.println("LOG (UI-SPINNER): Updating spinners from instance. UI events disabled.");

        // Position
        Vec3 pos = selectedInstance.getPosition();
        posXSpinner.setValue(pos.x);
        posYSpinner.setValue(pos.y);
        posZSpinner.setValue(pos.z);

        // Scale
        Vec3 scale = selectedInstance.getScale();
        uniformScaleSpinner.setValue(scale.x);

        // Color
        Vec3 color = selectedInstance.getColor();
        colorRSpinner.setValue(color.x);
        colorGSpinner.setValue(color.y);
        colorBSpinner.setValue(color.z);

        // Find matching Color preset
        boolean matchedColorPreset = false;
        for (int i = 0; i < colorComboBox.getItemCount(); i++) {
            ColorPreset preset = colorComboBox.getItemAt(i);
            Vec3 presetColor = preset.getColor();
            if (presetColor != null && presetColor.x == color.x && presetColor.y == color.y && presetColor.z == color.z) {
                colorComboBox.setSelectedIndex(i);
                rgbPanel.setVisible(false);
                matchedColorPreset = true;
                break;
            }
        }
        if (!matchedColorPreset) {
            for (int i = 0; i < colorComboBox.getItemCount(); i++) {
                if (colorComboBox.getItemAt(i).getName().equals("Custom...")) {
                    colorComboBox.setSelectedIndex(i);
                    rgbPanel.setVisible(true);
                    break;
                }
            }
        }

        // Update Material ComboBox
        float matType = selectedInstance.getMaterialType();
        boolean matchedMatPreset = false;
        for (int i = 0; i < materialComboBox.getItemCount(); i++) {
            MaterialPreset preset = materialComboBox.getItemAt(i);
            if (preset.getType() == matType) {
                materialComboBox.setSelectedIndex(i);
                matchedMatPreset = true;
                break;
            }
        }
        if (!matchedMatPreset) {
            materialComboBox.setSelectedIndex(0); // Default to first item
        }

        // System.out.println("LOG (UI-SPINNER): Spinners updated. UI events re-enabled.");
        isUpdatingUI = false; // Clear flag
    }

    /**
     * Set up Key Bindings (Q/E for Up/Down)
     */
    private void setupKeyBindings() {
        InputMap inputMap = imageLabel.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        ActionMap actionMap = imageLabel.getActionMap();

        class CameraAction extends AbstractAction {
            private final String key;
            private final Vec3 moveVector;
            CameraAction(String key, Vec3 moveVector) { this.key = key; this.moveVector = moveVector; }
            @Override
            public void actionPerformed(ActionEvent e) {
                camera.setOrigin(camera.getOrigin().add(moveVector));
                vulkanEngine.submitCamera(camera.state()); // the engine restarts accumulation
            }
        }

        // WASD
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_W, 0), "moveForward");
        actionMap.put("moveForward", new CameraAction("W", new Vec3(0, 0, -6.5)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_S, 0), "moveBackward");
        actionMap.put("moveBackward", new CameraAction("S", new Vec3(0, 0, 15.0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_A, 0), "moveLeft");
        actionMap.put("moveLeft", new CameraAction("A", new Vec3(-5.5, 0, 0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_D, 0), "moveRight");
        actionMap.put("moveRight", new CameraAction("D", new Vec3(5.5, 0, 0)));

        // Q/E (Up/Down)
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_Q, 0), "moveUp");
        actionMap.put("moveUp", new CameraAction("Q", new Vec3(0, 3.5, 0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_E, 0), "moveDown");
        actionMap.put("moveDown", new CameraAction("E", new Vec3(0, -3.5, 0)));
    }


    /** Sends new render settings to the engine (which restarts accumulation). */
    private void updateSettings(RenderSettings settings) {
        renderSettings = settings;
        vulkanEngine.submitSettings(settings);
    }
}