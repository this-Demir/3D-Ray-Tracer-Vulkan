package dev.demir.vulkan.ui;

import dev.demir.vulkan.scene.ModelInstance;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.util.function.Consumer;

/**
 * "Objects" box: the objects in the scene, with Duplicate and Remove.
 * The list model is the scene's source of truth (UI thread only).
 */
public final class ObjectListPanel extends JPanel {

    private final JList<ModelInstance> list;

    public ObjectListPanel(DefaultListModel<ModelInstance> objects, Consumer<ModelInstance> onSelect,
                           Consumer<ModelInstance> onDuplicate, Consumer<ModelInstance> onRemove) {
        super(new BorderLayout(4, 4));
        setBorder(new TitledBorder("Objects"));
        list = new JList<>(objects);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setVisibleRowCount(6);
        list.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) onSelect.accept(list.getSelectedValue());
        });

        JButton duplicateButton = new JButton("Duplicate");
        duplicateButton.addActionListener(e -> {
            if (list.getSelectedValue() != null) onDuplicate.accept(list.getSelectedValue());
        });
        JButton removeButton = new JButton("Remove");
        removeButton.addActionListener(e -> {
            if (list.getSelectedValue() != null) onRemove.accept(list.getSelectedValue());
        });
        JPanel buttons = new JPanel(new GridLayout(1, 2, 4, 0));
        buttons.add(duplicateButton);
        buttons.add(removeButton);

        add(new JScrollPane(list), BorderLayout.CENTER);
        add(buttons, BorderLayout.SOUTH);
    }

    public void select(ModelInstance instance) {
        list.setSelectedValue(instance, true);
    }

    public void refresh() {
        list.repaint();
    }
}
