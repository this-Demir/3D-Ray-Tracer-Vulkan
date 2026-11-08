package dev.demir.vulkan.ui;

import javax.swing.*;
import java.awt.*;

/** Small GridBagLayout helpers for label / field rows. */
final class Forms {

    private Forms() {
    }

    /** Adds "label | component" as row 'row' of a GridBagLayout panel. */
    static void row(JPanel panel, int row, String label, Component component) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridy = row;
        c.insets = new Insets(2, 4, 2, 4);
        c.gridx = 0;
        c.anchor = GridBagConstraints.WEST;
        JLabel jLabel = new JLabel(label);
        panel.add(jLabel, c);
        if (component instanceof JSpinner) {
            jLabel.setLabelFor(component);
        }

        c.gridx = 1;
        c.weightx = 1;
        c.fill = GridBagConstraints.HORIZONTAL;
        panel.add(component, c);
    }

    /** A spinner for decimal values with a compact editor. */
    static JSpinner decimalSpinner(double value, double min, double max, double step) {
        JSpinner spinner = new JSpinner(new SpinnerNumberModel(value, min, max, step));
        JSpinner.NumberEditor editor = new JSpinner.NumberEditor(spinner, "0.###");
        spinner.setEditor(editor);
        editor.getTextField().setColumns(5);
        return spinner;
    }

    /** Three spinners side by side (x, y, z). */
    static JPanel triple(JSpinner x, JSpinner y, JSpinner z) {
        JPanel panel = new JPanel(new GridLayout(1, 3, 3, 0));
        panel.add(x);
        panel.add(y);
        panel.add(z);
        return panel;
    }
}
