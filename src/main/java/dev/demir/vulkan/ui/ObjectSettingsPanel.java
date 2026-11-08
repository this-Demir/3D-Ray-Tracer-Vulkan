package dev.demir.vulkan.ui;

import dev.demir.vulkan.scene.Material;
import dev.demir.vulkan.scene.ModelInstance;
import dev.demir.vulkan.util.Vec3;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;

/**
 * "Object settings" box: name, position, rotation, scale and material of the selected object.
 * Every edit is applied to the object right away and reported through onChanged.
 */
public final class ObjectSettingsPanel extends JPanel {

    private final JTextField nameField = new JTextField();
    private final JSpinner posX = Forms.decimalSpinner(0, -100000, 100000, 1);
    private final JSpinner posY = Forms.decimalSpinner(0, -100000, 100000, 1);
    private final JSpinner posZ = Forms.decimalSpinner(0, -100000, 100000, 1);
    private final JSpinner rotX = Forms.decimalSpinner(0, -360, 360, 5);
    private final JSpinner rotY = Forms.decimalSpinner(0, -360, 360, 5);
    private final JSpinner rotZ = Forms.decimalSpinner(0, -360, 360, 5);
    private final JSpinner scaleX = Forms.decimalSpinner(1, 0.0001, 100000, 0.1);
    private final JSpinner scaleY = Forms.decimalSpinner(1, 0.0001, 100000, 0.1);
    private final JSpinner scaleZ = Forms.decimalSpinner(1, 0.0001, 100000, 0.1);
    private final JCheckBox uniformScale = new JCheckBox("Uniform", true);
    private final JComboBox<Material.Type> typeBox = new JComboBox<>(Material.Type.values());
    private final JButton colorButton = new JButton(" ");
    private final JSpinner roughness = Forms.decimalSpinner(0, 0, 1, 0.05);
    private final JSpinner emission = Forms.decimalSpinner(1, 0, 100000, 1);
    private final JSpinner ior = Forms.decimalSpinner(1.5, 1, 3, 0.05);

    private final Runnable onChanged;
    private ModelInstance selected;
    private boolean updating;

    public ObjectSettingsPanel(Runnable onChanged) {
        super(new GridBagLayout());
        this.onChanged = onChanged;
        setBorder(new TitledBorder("Object settings"));

        JPanel scaleRow = new JPanel(new BorderLayout(4, 0));
        scaleRow.add(Forms.triple(scaleX, scaleY, scaleZ), BorderLayout.CENTER);
        scaleRow.add(uniformScale, BorderLayout.EAST);

        colorButton.setToolTipText("Albedo / light color");
        roughness.setToolTipText("Metal: 0 = mirror, 1 = very blurry");
        emission.setToolTipText("Light: brightness multiplier");
        ior.setToolTipText("Glass: index of refraction (glass 1.5, water 1.33, diamond 2.4)");

        int row = 0;
        Forms.row(this, row++, "Name:", nameField);
        Forms.row(this, row++, "Position:", Forms.triple(posX, posY, posZ));
        Forms.row(this, row++, "Rotation:", Forms.triple(rotX, rotY, rotZ));
        Forms.row(this, row++, "Scale:", scaleRow);
        Forms.row(this, row++, "Material:", typeBox);
        Forms.row(this, row++, "Color:", colorButton);
        Forms.row(this, row++, "Roughness:", roughness);
        Forms.row(this, row++, "Emission:", emission);
        Forms.row(this, row, "IOR:", ior);

        nameField.addActionListener(e -> applyName());
        nameField.addFocusListener(new java.awt.event.FocusAdapter() {
            @Override
            public void focusLost(java.awt.event.FocusEvent e) {
                applyName();
            }
        });
        for (JSpinner spinner : new JSpinner[]{posX, posY, posZ, rotX, rotY, rotZ, roughness, emission, ior}) {
            spinner.addChangeListener(e -> apply());
        }
        for (JSpinner spinner : new JSpinner[]{scaleX, scaleY, scaleZ}) {
            spinner.addChangeListener(e -> {
                if (!updating && uniformScale.isSelected()) {
                    syncUniformScale((JSpinner) e.getSource());
                }
                apply();
            });
        }
        typeBox.addActionListener(e -> apply());
        colorButton.addActionListener(e -> chooseColor());
        show(null);
    }

    /** Shows the given object (or disables the panel when null). */
    public void show(ModelInstance instance) {
        selected = instance;
        updating = true;
        try {
            boolean enabled = instance != null;
            for (Component c : getComponents()) setEnabledDeep(c, enabled);
            if (!enabled) return;
            nameField.setText(instance.getDisplayName());
            set(posX, posY, posZ, instance.getPosition());
            set(rotX, rotY, rotZ, instance.getRotation());
            set(scaleX, scaleY, scaleZ, instance.getScale());
            Vec3 s = instance.getScale();
            uniformScale.setSelected(s.x == s.y && s.y == s.z);
            Material m = instance.getMaterial();
            typeBox.setSelectedItem(m.type());
            roughness.setValue((double) m.roughness());
            emission.setValue((double) m.emission());
            ior.setValue((double) m.ior());
            colorButton.setBackground(toColor(m.albedo()));
            updateEnabledFields(m.type());
        } finally {
            updating = false;
        }
    }

    private void apply() {
        if (updating || selected == null) return;
        selected.setPosition(get(posX, posY, posZ));
        selected.setRotation(get(rotX, rotY, rotZ));
        selected.setScale(get(scaleX, scaleY, scaleZ));
        Material.Type type = (Material.Type) typeBox.getSelectedItem();
        Material m = selected.getMaterial().withType(type)
                .withRoughness(((Number) roughness.getValue()).floatValue())
                .withEmission(((Number) emission.getValue()).floatValue())
                .withIor(((Number) ior.getValue()).floatValue());
        selected.setMaterial(m);
        updateEnabledFields(type);
        onChanged.run();
    }

    private void applyName() {
        if (updating || selected == null) return;
        String name = nameField.getText().trim();
        if (!name.isEmpty() && !name.equals(selected.getDisplayName())) {
            selected.setDisplayName(name);
            onChanged.run();
        }
    }

    private void chooseColor() {
        if (selected == null) return;
        Color chosen = JColorChooser.showDialog(this, "Color", toColor(selected.getMaterial().albedo()));
        if (chosen != null) {
            selected.setMaterial(selected.getMaterial().withAlbedo(
                    chosen.getRed() / 255f, chosen.getGreen() / 255f, chosen.getBlue() / 255f));
            colorButton.setBackground(chosen);
            onChanged.run();
        }
    }

    private void syncUniformScale(JSpinner source) {
        updating = true;
        try {
            Object value = source.getValue();
            scaleX.setValue(value);
            scaleY.setValue(value);
            scaleZ.setValue(value);
        } finally {
            updating = false;
        }
    }

    private void updateEnabledFields(Material.Type type) {
        roughness.setEnabled(type == Material.Type.METAL);
        emission.setEnabled(type == Material.Type.EMISSIVE);
        ior.setEnabled(type == Material.Type.GLASS);
    }

    private static void setEnabledDeep(Component component, boolean enabled) {
        component.setEnabled(enabled);
        if (component instanceof Container container) {
            for (Component child : container.getComponents()) setEnabledDeep(child, enabled);
        }
    }

    private static void set(JSpinner x, JSpinner y, JSpinner z, Vec3 v) {
        x.setValue(v.x);
        y.setValue(v.y);
        z.setValue(v.z);
    }

    private static Vec3 get(JSpinner x, JSpinner y, JSpinner z) {
        return new Vec3(((Number) x.getValue()).doubleValue(), ((Number) y.getValue()).doubleValue(),
                ((Number) z.getValue()).doubleValue());
    }

    private static Color toColor(float[] rgb) {
        return new Color(clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]));
    }

    private static float clamp(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}
