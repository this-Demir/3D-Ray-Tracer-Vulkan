package dev.demir.vulkan;

import dev.demir.vulkan.engine.VulkanEngine;
import dev.demir.vulkan.renderer.FrameData;
import dev.demir.vulkan.renderer.SceneBuilder;
import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.scene.ModelInstance;
import dev.demir.vulkan.scene.Scene;
import dev.demir.vulkan.util.Vec3;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.Objects;
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
    private byte[] swizzleBuffer;

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
    private final SceneBuilder sceneBuilder = new SceneBuilder();
    private final Camera camera;
    private final Scene scene = new Scene();

    // --- State ---
    private long lastFrameTime = System.nanoTime();
    private int frameCount = 0;

    // --- 3-THREAD-FIX: This flag pauses accumulation in updateUI() ---
    private volatile boolean sceneBuildInProgress = false;

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
        this.vulkanEngine = new VulkanEngine(latestFrame);
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
        frame = new JFrame("Vulkan BVH Ray Tracer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        bufferedImage = new BufferedImage(RENDER_WIDTH, RENDER_HEIGHT, BufferedImage.TYPE_3BYTE_BGR);
        imageLabel = new JLabel(new ImageIcon(bufferedImage));
        frame.getContentPane().add(imageLabel, BorderLayout.CENTER);

        // 1b. Set up the Scene Control Panel
        setupSceneControls();
        frame.getContentPane().add(controlPanel, BorderLayout.EAST);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        swizzleBuffer = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();

        // 2. Start VulkanEngine Thread (VRT)
        System.out.println("LOG (UI-RUN): Starting VulkanEngine...");
        vulkanEngine.start();

        // 3. Trigger initial scene build (SRT)
        populateDefaultScene();
        rebuildSceneAsync(); // This will also call resetAccumulation

        // 4. Send initial camera state to the engine
        // (This is now handled by rebuildSceneAsync's .whenComplete block)
        System.out.println("LOG (UI-RUN): Waiting for first scene build...");

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
     * Main UI loop driven by the Swing Timer.
     * (3-THREAD-FIX): This is now the *only* place frameCount is incremented.
     * It does *not* send sky state.
     */
    private void updateUI() {
        String logPrefix = "LOG (UI-TIMER): ";

        if (vulkanEngine != null) {

            // --- 3-THREAD-FIX: PAUSE ACCUMULATION ---
            // Only increment the frame counter if a scene rebuild (SRT)
            // is not in progress. This ensures that when the SRT
            // finishes, the frameCount is still 0.
            if (!sceneBuildInProgress) {
                camera.incrementFrameCount();
                // System.out.println(logPrefix + "Accumulation ON. Incremented frameCount to " + camera.getFrameCount());
            } else {
                // System.out.println(logPrefix + "Accumulation PAUSED (SRT running). frameCount is " + camera.getFrameCount());
            }

            // --- 3-THREAD-FIX: REMOVE OVERRIDE ---
            // DO NOT send sky state here. This was overriding the
            // user's selection from the ActionListener.
            // vulkanEngine.submitSkyToggle(enableSkyCheckBox.isSelected()); // <-- BUG WAS HERE

            // Always send the latest camera state (with the new or paused frameCount)
            // This drives the continuous accumulation.
            vulkanEngine.submitCameraUpdate(camera);
        }

        FrameData frameData = latestFrame.getAndSet(null);
        if (frameData != null) {
            updateBufferedImage(frameData.pixelData);
            imageLabel.repaint();
            frameCount++;
        }
        long now = System.nanoTime();
        if (now - lastFrameTime >= 1_000_000_000) {
            // Update FPS counter and also show accumulated frames
            int accumulatedFrames = (camera != null) ? camera.getFrameCount() : 0;
            // System.out.println(logPrefix + "FPS Update. Samples: " + accumulatedFrames);
            frame.setTitle(String.format("Vulkan Ray Tracer | %d FPS | Samples: %d", frameCount, accumulatedFrames));
            frameCount = 20;
            lastFrameTime = now;
        }
    }

    /**
     * Assigns a scene build task to the SceneBuilder (SRT).
     * (3-THREAD-FIX): This now controls the sceneBuildInProgress flag
     * and handles resetting the camera *after* the build.
     */
    public void rebuildSceneAsync() {
        String logPrefix = "LOG (UI-REBUILD): ";

        if (sceneBuildInProgress) {
            System.out.println(logPrefix + "Scene build already in progress. Ignoring trigger.");
            return;
        }

        // 1. Set flag to TRUE to pause accumulation in updateUI()
        sceneBuildInProgress = true;

        System.out.println(logPrefix + "Triggering asynchronous scene build (SRT)...");
        System.out.println(logPrefix + "Accumulation is PAUSED.");

        // --- 3-THREAD-FIX: MOVE RESET TO END ---
        // Do NOT reset the camera here. If we do, updateUI()
        // will increment frameCount while the SRT is running.
        // We move the reset logic into the .whenCompleteAsync block.
        // --- END FIX ---


        final Scene sceneSnapshot = scene.createSnapshot();
        CompletableFuture
                .supplyAsync(() -> sceneBuilder.buildScene(sceneSnapshot))
                .whenCompleteAsync((builtCpuData, error) -> {
                    // This block runs on the UI Thread (thanks to SwingUtilities::invokeLater)
                    String logPrefixDone = "LOG (UI-SRT-DONE): ";

                    if (error != null) {
                        error.printStackTrace();
                        JOptionPane.showMessageDialog(frame, "Scene build failed: " + error.getMessage(), "SRT Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        System.out.println(logPrefixDone + "SRT finished. Sending new scene to VRT...");

                        // 1. Send the new scene data to the VRT.
                        //    (VulkanEngine's submitScene is now "dumb" and doesn't reset)
                        vulkanEngine.submitScene(builtCpuData);
                        System.out.println(logPrefixDone + "submitScene() called.");


                        // --- 3-THREAD-FIX: RESET CAMERA *AFTER* BUILD ---
                        // 2. Now that the scene is submitted, reset the camera
                        //    (frameCount = 0) on the UI thread.
                        camera.resetAccumulation();
                        System.out.println(logPrefixDone + "camera.resetAccumulation() called. frameCount is now 0.");

                        // 3. Send the current sky state AND the reset (frameCount=0)
                        //    to the VRT. This ensures the VRT's *next* frame
                        //    is the new scene with frame 0.
                        boolean isSkyEnabled = enableSkyCheckBox.isSelected();
                        vulkanEngine.submitSkyToggle(isSkyEnabled);
                        vulkanEngine.submitCameraUpdate(camera);
                        System.out.println(logPrefixDone + "submitSkyToggle(" + isSkyEnabled + ") and submitCameraUpdate(fc=0) called.");
                        // --- END FIX ---
                    }

                    // 4. Finally, set flag to FALSE to re-enable
                    //    accumulation (incrementFrameCount) in updateUI().
                    sceneBuildInProgress = false;
                    System.out.println(logPrefixDone + "Accumulation is RESUMED.");

                }, SwingUtilities::invokeLater);
    }

    /**
     * Loads the default scene ('ground_plane.obj', 'car.obj', 'sun.obj')
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

        // --- CAR (Metal) ---
        ModelInstance car = new ModelInstance("./objects/car.obj", "Car");
        car.setPosition(new Vec3(0, -8, 0));
        car.setScale(new Vec3(2, 2, 2));
        car.setColor(new Vec3(0.6, 0.7, 0.1));
        car.setMaterialType(1.0f); // 1.0f = Metal
        scene.addInstance(car);
        listModel.addElement(car);

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


        sceneObjectList.setSelectedValue(car, true); // Select the car by default
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
        buttonPanel.add(addButton);
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
        enableSkyCheckBox.addActionListener(e -> {
            // This is an "event". It resets accumulation.
            boolean isSkyEnabled = enableSkyCheckBox.isSelected(); // Get the new state
            System.out.println("LOG (UI-EVENT-SKY): Sky CheckBox clicked. Sending state: " + isSkyEnabled);

            // 1. Send the new state
            vulkanEngine.submitSkyToggle(isSkyEnabled);

            // 2. Reset the counter
            camera.resetAccumulation();
            System.out.println("LOG (UI-EVENT-SKY): camera.resetAccumulation() called. frameCount is now 0.");

            // 3. Send the reset (frameCount=0)
            // !!! BU SATIR BÜYÜK İHTİMALLE EKSİKTİ !!!
            vulkanEngine.submitCameraUpdate(camera);
            System.out.println("LOG (UI-EVENT-SKY): submitCameraUpdate(fc=0) called.");
        });
        addLabelAndComponent(panel, "", enableSkyCheckBox, gbc, 0); // No label

        // --- Exposure (Composure) Slider ---
        exposureSlider = new JSlider(JSlider.HORIZONTAL, -50, 50, 0); // Range -5.0 to +5.0 (x10)
        exposureSlider.setMajorTickSpacing(25);
        exposureSlider.setMinorTickSpacing(5);
        exposureSlider.setPaintTicks(true);
        exposureSlider.setPaintLabels(false); // Labels would be -50, 0, 50. Not user-friendly.

        exposureSlider.addChangeListener(e -> {
            if (!exposureSlider.getValueIsAdjusting()) {
                // This is an "event". It resets accumulation.
                float exposureValue = exposureSlider.getValue() / 10.0f;
                System.out.println("LOG (UI-EVENT-EXPOSURE): Exposure set to: " + exposureValue);

                // --- TODO: Send exposure value to VulkanEngine ---
                // vulkanEngine.submitExposureUpdate(exposureValue);

                camera.resetAccumulation(); // 1. Reset the counter
                System.out.println("LOG (UI-EVENT-EXPOSURE): camera.resetAccumulation() called. frameCount is now 0.");
                // 2. Send the *current* sky state along with the reset
                boolean isSkyEnabled = enableSkyCheckBox.isSelected();
                vulkanEngine.submitSkyToggle(isSkyEnabled);
                vulkanEngine.submitCameraUpdate(camera); // 3. Send the reset (frameCount=0)
                System.out.println("LOG (UI-EVENT-EXPOSURE): submitSkyToggle(" + isSkyEnabled + ") and submitCameraUpdate(fc=0) called.");
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
                new MaterialPreset("Emissive (Light)", 3.0f)
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
                String logPrefix = "LOG (UI-EVENT-KEY): ";
                System.out.println(logPrefix + "Key '" + key + "' pressed.");

                camera.setOrigin(camera.getOrigin().add(moveVector));

                // This is an "event". It resets accumulation.
                camera.resetAccumulation(); // 1. Reset the counter
                System.out.println(logPrefix + "camera.resetAccumulation() called. frameCount is now 0.");

                // 2. Send the *current* sky state along with the reset
                boolean isSkyEnabled = enableSkyCheckBox.isSelected();
                vulkanEngine.submitSkyToggle(isSkyEnabled);

                // 3. Send the reset (frameCount=0)
                vulkanEngine.submitCameraUpdate(camera);
                System.out.println(logPrefix + "submitSkyToggle(" + isSkyEnabled + ") and submitCameraUpdate(fc=0) called.");
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


    /**
     * Copies Vulkan's ByteBuffer (RGBA) into Swing's BufferedImage (BGR).
     */
    private void updateBufferedImage(ByteBuffer pixelData) {
        if (pixelData.capacity() < (RENDER_WIDTH * RENDER_HEIGHT * 4)) {
            // Can happen if the scene is empty (returns 1-byte buffer)
            return;
        }
        pixelData.rewind();
        for (int i = 0; i < (RENDER_WIDTH * RENDER_HEIGHT); i++) {
            // Vulkan (RGBA) -> Swing (BGR)
            swizzleBuffer[i * 3 + 0] = pixelData.get(i * 4 + 2); // Blue
            swizzleBuffer[i * 3 + 1] = pixelData.get(i * 4 + 1); // Green
            swizzleBuffer[i * 3 + 2] = pixelData.get(i * 4 + 0); // Red
        }
    }
}