package dev.demir.vulkan.ui;

import dev.demir.vulkan.scene.Camera;
import dev.demir.vulkan.util.Vec3;

import javax.swing.*;
import java.awt.event.*;

/**
 * Mouse and keyboard camera control on the viewport (UI thread):
 * - drag with the left mouse button: orbit around the look-at point
 * - mouse wheel: zoom in / out
 * - W/S: move forward / back, A/D: left / right, Q/E: up / down (moves camera and target together)
 * Every change is reported through onChange so the renderer gets the new camera.
 */
public final class CameraController {

    private final Camera camera;
    private final Runnable onChange;
    private int lastX, lastY;

    private CameraController(Camera camera, Runnable onChange) {
        this.camera = camera;
        this.onChange = onChange;
    }

    public static void attach(JComponent viewport, Camera camera, Runnable onChange) {
        CameraController controller = new CameraController(camera, onChange);
        viewport.setToolTipText("Drag: orbit  |  Wheel: zoom  |  W/A/S/D/Q/E: move");
        MouseAdapter mouse = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                controller.lastX = e.getX();
                controller.lastY = e.getY();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    controller.orbit(e.getX() - controller.lastX, e.getY() - controller.lastY);
                }
                controller.lastX = e.getX();
                controller.lastY = e.getY();
            }

            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                controller.zoom(e.getPreciseWheelRotation());
            }
        };
        viewport.addMouseListener(mouse);
        viewport.addMouseMotionListener(mouse);
        viewport.addMouseWheelListener(mouse);

        InputMap inputMap = viewport.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        ActionMap actionMap = viewport.getActionMap();
        bind(inputMap, actionMap, KeyEvent.VK_W, () -> controller.move(1, 0, 0));
        bind(inputMap, actionMap, KeyEvent.VK_S, () -> controller.move(-1, 0, 0));
        bind(inputMap, actionMap, KeyEvent.VK_D, () -> controller.move(0, 1, 0));
        bind(inputMap, actionMap, KeyEvent.VK_A, () -> controller.move(0, -1, 0));
        bind(inputMap, actionMap, KeyEvent.VK_Q, () -> controller.move(0, 0, 1));
        bind(inputMap, actionMap, KeyEvent.VK_E, () -> controller.move(0, 0, -1));
    }

    private static void bind(InputMap inputMap, ActionMap actionMap, int key, Runnable action) {
        String name = "camera-" + key;
        inputMap.put(KeyStroke.getKeyStroke(key, 0), name);
        actionMap.put(name, new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                action.run();
            }
        });
    }

    /** Rotates the camera around its look-at point (yaw around world Y, pitch clamped short of the poles). */
    private void orbit(int dx, int dy) {
        Vec3 offset = camera.getOrigin().sub(camera.getLookAt());
        double radius = offset.length();
        double yaw = Math.atan2(offset.x, offset.z) - dx * 0.008;
        double pitch = Math.asin(offset.y / radius) + dy * 0.008;
        pitch = Math.max(-1.5, Math.min(1.5, pitch));
        Vec3 newOffset = new Vec3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch))
                .multiply(radius);
        camera.setOrigin(camera.getLookAt().add(newOffset));
        onChange.run();
    }

    private void zoom(double notches) {
        Vec3 offset = camera.getOrigin().sub(camera.getLookAt());
        double factor = Math.pow(1.12, notches);
        if (offset.length() * factor < 0.5) return;
        camera.setOrigin(camera.getLookAt().add(offset.multiply(factor)));
        onChange.run();
    }

    /** Moves camera and target together; the step grows with the distance to the target. */
    private void move(int forward, int right, int up) {
        Vec3 view = camera.getLookAt().sub(camera.getOrigin());
        double step = Math.max(0.5, view.length() * 0.05);
        Vec3 forwardDir = new Vec3(view.x, 0, view.z);
        forwardDir = forwardDir.length() < 1e-6 ? new Vec3(0, 0, -1) : forwardDir.unit();
        Vec3 rightDir = Vec3.cross(forwardDir, new Vec3(0, 1, 0));
        Vec3 delta = forwardDir.multiply(forward * step)
                .add(rightDir.multiply(right * step))
                .add(new Vec3(0, up * step, 0));
        camera.setOrigin(camera.getOrigin().add(delta));
        camera.setLookAt(camera.getLookAt().add(delta));
        onChange.run();
    }
}
