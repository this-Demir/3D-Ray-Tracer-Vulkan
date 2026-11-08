package dev.demir.vulkan.ui;

import dev.demir.vulkan.scene.Material;
import dev.demir.vulkan.scene.ModelInstance;
import dev.demir.vulkan.scene.ScenePresets;
import dev.demir.vulkan.util.Vec3;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * "Object pool" box: things that can be added to the scene, i.e. the built-in sphere and the
 * .obj models in the objects/ folder. Double-click or "Add" places a new object at the
 * point the camera looks at; "Browse..." adds any .obj file from disk.
 */
public final class ObjectPoolPanel extends JPanel {

    /** One entry in the pool, with a starting size and rotation that make it look reasonable. */
    private record Entry(String name, String path, double scale, Vec3 rotation) {
        @Override
        public String toString() {
            return name;
        }
    }

    /** Starting transforms for the bundled models (they come in very different sizes). */
    private static final Map<String, Entry> KNOWN = Map.of(
            "FinalBaseMesh.obj", new Entry("Human", ScenePresets.HUMAN, 0.5, Vec3.ZERO),
            "plane.obj", new Entry("Airplane", ScenePresets.AIRPLANE, 0.016, new Vec3(-90, 0, 0)),
            "Moon.obj", new Entry("Moon (mesh)", ScenePresets.MOON, 2, Vec3.ZERO),
            "cube.obj", new Entry("Cube", ScenePresets.CUBE, 5, Vec3.ZERO),
            "ground_plane.obj", new Entry("Quad (floor / wall)", ScenePresets.QUAD, 10, Vec3.ZERO),
            "white_mesh.obj", new Entry("Sculpture", ScenePresets.BLOB, 3, Vec3.ZERO)
    );

    private final DefaultListModel<Entry> entries = new DefaultListModel<>();
    private final JList<Entry> list = new JList<>(entries);
    private final Consumer<ModelInstance> onAdd;
    private final Supplier<Vec3> placement;

    public ObjectPoolPanel(Supplier<Vec3> placement, Consumer<ModelInstance> onAdd) {
        super(new BorderLayout(4, 4));
        this.onAdd = onAdd;
        this.placement = placement;
        setBorder(new TitledBorder("Object pool"));

        entries.addElement(new Entry("Sphere", ScenePresets.SPHERE, 3, Vec3.ZERO));
        File[] files = new File("objects").listFiles((dir, name) -> name.toLowerCase().endsWith(".obj"));
        if (files != null) {
            Arrays.sort(files, Comparator.comparing(File::getName));
            for (File file : files) {
                Entry known = KNOWN.get(file.getName());
                entries.addElement(known != null ? known
                        : new Entry(file.getName(), "./objects/" + file.getName(), 1, Vec3.ZERO));
            }
        }

        list.setVisibleRowCount(5);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setSelectedIndex(0);
        list.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) addSelected();
            }
        });

        JButton addButton = new JButton("Add");
        addButton.addActionListener(e -> addSelected());
        JButton browseButton = new JButton("Browse...");
        browseButton.addActionListener(e -> browse());
        JPanel buttons = new JPanel(new GridLayout(1, 2, 4, 0));
        buttons.add(addButton);
        buttons.add(browseButton);

        add(new JScrollPane(list), BorderLayout.CENTER);
        add(buttons, BorderLayout.SOUTH);
    }

    private void addSelected() {
        Entry entry = list.getSelectedValue();
        if (entry != null) add(entry);
    }

    private void add(Entry entry) {
        Vec3 at = placement.get();
        ModelInstance instance = new ModelInstance(entry.path(), entry.name())
                .at(at.x, at.y, at.z)
                .rotated(entry.rotation().x, entry.rotation().y, entry.rotation().z)
                .scaled(entry.scale())
                .with(Material.diffuse(0.8f, 0.8f, 0.8f));
        onAdd.accept(instance);
    }

    private void browse() {
        JFileChooser chooser = new JFileChooser(".");
        chooser.setDialogTitle("Add an OBJ model");
        chooser.setFileFilter(new FileNameExtensionFilter("OBJ models (*.obj)", "obj"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            add(new Entry(file.getName(), file.getPath(), 1, Vec3.ZERO));
        }
    }
}
