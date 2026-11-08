package dev.demir.vulkan.ui;

import javax.swing.*;
import java.awt.*;

/**
 * Vertical stack of panels for a JScrollPane that always matches the scroll pane's width
 * (only scrolls vertically), so fields are never cut off on the right.
 */
public final class SideColumn extends JPanel implements Scrollable {

    public SideColumn(JComponent... panels) {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        for (JComponent panel : panels) {
            panel.setAlignmentX(Component.LEFT_ALIGNMENT);
            add(panel);
        }
        add(Box.createVerticalGlue());
    }

    @Override
    public Dimension getPreferredScrollableViewportSize() {
        return getPreferredSize();
    }

    @Override
    public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 16;
    }

    @Override
    public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction) {
        return visibleRect.height - 32;
    }

    @Override
    public boolean getScrollableTracksViewportWidth() {
        return true;
    }

    @Override
    public boolean getScrollableTracksViewportHeight() {
        return false;
    }
}
