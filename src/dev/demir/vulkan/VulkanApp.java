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
import javax.swing.event.ChangeEvent; // NEW Import
import javax.swing.event.ChangeListener; // NEW Import
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
        System.out.println("LOG (UI): Starting VulkanEngine...");
        vulkanEngine.start();

        // 3. Trigger initial scene build (SRT)
        populateDefaultScene();
        rebuildSceneAsync(); // This will also call resetAccumulation

        // 4. Send initial camera state to the engine
        camera.resetAccumulation(); // Ensure first frame is clean
        vulkanEngine.submitCameraUpdate(camera);

        // 5. Start UI Timer
        Timer swingTimer = new Timer(16, (e) -> updateUI());
        swingTimer.start();

        // 6. Set up keyboard controls
        setupKeyBindings();

        // 7. Handle window closing
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                System.out.println("LOG (UI): Stopping engine...");
                swingTimer.stop();
                vulkanEngine.stop();
            }
        });
    }

    /**
     * Main UI loop driven by the Swing Timer.
     */
    private void updateUI() {
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
            frame.setTitle(String.format("Vulkan Ray Tracer | %d FPS | Samples: %d", frameCount, accumulatedFrames));
            frameCount = 0;
            lastFrameTime = now;
        }
    }

    /**
     * Assigns a scene build task to the SceneBuilder (SRT).
     */
    public void rebuildSceneAsync() {
        if (sceneBuildInProgress) {
            System.out.println("LOG (UI): Scene build already in progress. Ignoring trigger.");
            return;
        }
        sceneBuildInProgress = true;
        System.out.println("LOG (UI): Triggering asynchronous scene build (SRT)...");

        // FIX: Reset accumulation whenever the scene is rebuilt
        camera.resetAccumulation();
        vulkanEngine.submitCameraUpdate(camera); // Send reset to engine

        final Scene sceneSnapshot = scene.createSnapshot();
        CompletableFuture
                .supplyAsync(() -> sceneBuilder.buildScene(sceneSnapshot))
                .whenCompleteAsync((builtCpuData, error) -> {
                    if (error != null) {
                        error.printStackTrace();
                        JOptionPane.showMessageDialog(frame, "Scene build failed: " + error.getMessage(), "SRT Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        System.out.println("LOG (UI): SRT finished. Sending new scene to engine.");
                        vulkanEngine.submitScene(builtCpuData);
                    }
                    sceneBuildInProgress = false;
                }, SwingUtilities::invokeLater);
    }

    /**
     * Loads the default scene ('ground_plane.obj', 'car.obj', 'sun.obj')
     */
    private void populateDefaultScene() {
        System.out.println("LOG (UI): Populating default scene...");

        // --- GROUND (Matte) ---
        ModelInstance plane = new ModelInstance("ground_plane.obj", "Ground Plane");
        plane.setPosition(new Vec3(0, -10, 0));
        plane.setScale(new Vec3(150, 1, 150)); // Non-uniform scale
        plane.setColor(new Vec3(0.5, 0.5, 0.5)); // "Grey" preset
        plane.setMaterialType(0.0f); // 0.0f = Lambertian (Matte)
        scene.addInstance(plane);
        listModel.addElement(plane);

        // --- CAR (Metal) ---
        ModelInstance car = new ModelInstance("car.obj", "Car");
        car.setPosition(new Vec3(0, -8, 0));
        car.setScale(new Vec3(2, 2, 2));
        car.setColor(new Vec3(0.6, 0.7, 0.1));
        car.setMaterialType(1.0f); // 1.0f = Metal
        scene.addInstance(car);
        listModel.addElement(car);

        // --- LIGHT SOURCE (Emissive) ---
        ModelInstance light = new ModelInstance("sun.obj", "Light Source");
        light.setPosition(new Vec3(0, 220, 0));
        light.setScale(new Vec3(0.35, 0.35, 0.35));
        light.setColor(new Vec3(4.0, 4.0, 4.0)); // Bright white ( > 1.0)
        light.setMaterialType(3.0f); // 3.0f = Emissive
        scene.addInstance(light);
        listModel.addElement(light);


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
            boolean isSkyEnabled = enableSkyCheckBox.isSelected();
            vulkanEngine.submitSkyToggle(isSkyEnabled);
            camera.resetAccumulation();
            vulkanEngine.submitCameraUpdate(camera);
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
                // Only update when the user releases the slider
                float exposureValue = exposureSlider.getValue() / 10.0f;
                System.out.println("LOG (UI): Exposure set to: " + exposureValue);

                // --- TODO: Send exposure value to VulkanEngine ---
                // We need a new UBO field and submit method for this.
                // vulkanEngine.submitExposureUpdate(exposureValue);

                camera.resetAccumulation();
                vulkanEngine.submitCameraUpdate(camera);
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
            applyChangesToInstance();
            camera.resetAccumulation(); // Reset noise
            rebuildSceneAsync();
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
            applyChangesToInstance();
            camera.resetAccumulation();
            rebuildSceneAsync();
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
            File file = fileChooser.getSelectedFile();
            ModelInstance instance = new ModelInstance(file.getPath(), file.getName());
            scene.addInstance(instance);
            listModel.addElement(instance);
            sceneObjectList.setSelectedValue(instance, true);
            rebuildSceneAsync(); // This will reset accumulation
        }
    }

    /**
     * Action for the "Remove" button.
     */
    private void removeSelectedInstance() {
        if (selectedInstance != null) {
            System.out.println("LOG (UI): Removing model: " + selectedInstance.getDisplayName());
            scene.removeInstance(selectedInstance);
            listModel.removeElement(selectedInstance);
            selectedInstance = null;
            rebuildSceneAsync(); // This will reset accumulation
        }
    }


    /**
     * Action for the "Apply Changes" button.
     * Reads values from spinners and saves them to the selected ModelInstance.
     */
    private void applyChangesToInstance() {
        if (selectedInstance == null || isUpdatingUI) return;
        System.out.println("LOG (UI): Applying changes to " + selectedInstance.getDisplayName());

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

        isUpdatingUI = false; // Clear flag
    }

    /**
     * Set up Key Bindings (Q/E for Up/Down)
     */
    private void setupKeyBindings() {
        InputMap inputMap = imageLabel.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        ActionMap actionMap = imageLabel.getActionMap();

        class CameraAction extends AbstractAction {
            private final Vec3 moveVector;
            CameraAction(Vec3 moveVector) { this.moveVector = moveVector; }
            @Override
            public void actionPerformed(ActionEvent e) {
                camera.setOrigin(camera.getOrigin().add(moveVector));

                // FIX: Reset accumulation on *every* key press
                camera.resetAccumulation();

                vulkanEngine.submitCameraUpdate(camera);
            }
        }

        // WASD
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_W, 0), "moveForward");
        actionMap.put("moveForward", new CameraAction(new Vec3(0, 0, -6.5)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_S, 0), "moveBackward");
        actionMap.put("moveBackward", new CameraAction(new Vec3(0, 0, 15.0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_A, 0), "moveLeft");
        actionMap.put("moveLeft", new CameraAction(new Vec3(-5.5, 0, 0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_D, 0), "moveRight");
        actionMap.put("moveRight", new CameraAction(new Vec3(5.5, 0, 0)));

        // Q/E (Up/Down)
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_Q, 0), "moveUp");
        actionMap.put("moveUp", new CameraAction(new Vec3(0, 3.5, 0)));
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_E, 0), "moveDown");
        actionMap.put("moveDown", new CameraAction(new Vec3(0, -3.5, 0)));
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