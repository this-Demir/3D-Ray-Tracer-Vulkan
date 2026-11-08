package dev.demir.vulkan.ui;

import dev.demir.vulkan.scene.RenderSettings;
import dev.demir.vulkan.scene.ScenePreset;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.util.List;
import java.util.function.Consumer;

/**
 * "Scene" box: pick a preset scene and edit the global render settings
 * (sky light, maximum bounces, exposure, sample limit).
 */
public final class SceneSettingsPanel extends JPanel {

    private final JComboBox<ScenePreset> presetBox;
    private final JCheckBox skyBox = new JCheckBox("Sky light");
    private final JCheckBox animateBox = new JCheckBox("Play animation (orbits, spins)");
    private final JSpinner bouncesSpinner = new JSpinner(new SpinnerNumberModel(10, 1, 64, 1));
    private final JSlider exposureSlider = new JSlider(-50, 50, 0); // tenths of a stop
    private final JLabel exposureLabel = new JLabel("0.0 EV");
    private final JSpinner samplesSpinner = new JSpinner(new SpinnerNumberModel(8192, 0, 1_000_000, 1024));
    private final Consumer<RenderSettings> onSettingsChanged;
    private boolean updating;

    public SceneSettingsPanel(List<ScenePreset> presets, Consumer<ScenePreset> onLoadPreset,
                              Consumer<RenderSettings> onSettingsChanged) {
        super(new GridBagLayout());
        this.onSettingsChanged = onSettingsChanged;
        setBorder(new TitledBorder("Scene"));

        presetBox = new JComboBox<>(presets.toArray(new ScenePreset[0]));
        JButton loadButton = new JButton("Load");
        loadButton.addActionListener(e -> onLoadPreset.accept((ScenePreset) presetBox.getSelectedItem()));

        JPanel presetRow = new JPanel(new BorderLayout(4, 0));
        presetRow.add(presetBox, BorderLayout.CENTER);
        presetRow.add(loadButton, BorderLayout.EAST);

        JPanel exposureRow = new JPanel(new BorderLayout(4, 0));
        exposureRow.add(exposureSlider, BorderLayout.CENTER);
        exposureLabel.setPreferredSize(new Dimension(52, exposureLabel.getPreferredSize().height));
        exposureRow.add(exposureLabel, BorderLayout.EAST);

        samplesSpinner.setToolTipText("Stop after this many samples per pixel (0 = never stop)");
        bouncesSpinner.setToolTipText("Maximum light bounces per path (mirror rooms need many)");

        Forms.row(this, 0, "Preset:", presetRow);
        Forms.row(this, 1, "", skyBox);
        Forms.row(this, 2, "Bounces:", bouncesSpinner);
        Forms.row(this, 3, "Exposure:", exposureRow);
        Forms.row(this, 4, "Samples:", samplesSpinner);
        Forms.row(this, 5, "", animateBox);

        skyBox.addActionListener(e -> fireChanged());
        animateBox.addActionListener(e -> fireChanged());
        bouncesSpinner.addChangeListener(e -> fireChanged());
        samplesSpinner.addChangeListener(e -> fireChanged());
        exposureSlider.addChangeListener(e -> {
            exposureLabel.setText(String.format("%.1f EV", exposureSlider.getValue() / 10.0));
            if (!exposureSlider.getValueIsAdjusting()) fireChanged();
        });
    }

    /** Shows the given settings without reporting a change. */
    public void show(RenderSettings settings, ScenePreset preset) {
        updating = true;
        try {
            if (preset != null) presetBox.setSelectedItem(preset);
            skyBox.setSelected(settings.skyEnabled());
            bouncesSpinner.setValue(settings.maxBounces());
            exposureSlider.setValue(Math.round(settings.exposure() * 10));
            exposureLabel.setText(String.format("%.1f EV", settings.exposure()));
            samplesSpinner.setValue(settings.maxSamples());
            animateBox.setSelected(settings.animate());
        } finally {
            updating = false;
        }
    }

    private void fireChanged() {
        if (updating) return;
        onSettingsChanged.accept(new RenderSettings(skyBox.isSelected(), exposureSlider.getValue() / 10f,
                (Integer) bouncesSpinner.getValue(), (Integer) samplesSpinner.getValue(), animateBox.isSelected()));
    }
}
