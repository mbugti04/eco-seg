from pathlib import Path

from PyQt5.QtCore import (
    Qt,
    pyqtSlot,
    QPoint,
    QRectF,
    QPointF,
    QSizeF,
)
from PyQt5.QtGui import (
    QColor,
    QPixmap,
    QMouseEvent,
    QWheelEvent,
    QBrush,
    QPainter,
)
from PyQt5.QtWidgets import QFrame, QGraphicsView

from .graphics_scene import GraphicsScene


class GraphicsView(QGraphicsView):
    def __init__(self, brush_feedback, parent=None):
        super().__init__(parent)
        self._scene = GraphicsScene(self)
        self._pan_mode = False
        self._last_pos = QPoint()
        self._brush_feedback = brush_feedback
        self._sam_mode = False

        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))
        self.setFrameShape(QFrame.Shape.NoFrame)  # removes white widget outline
        self.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.BlankCursor)

        self.image_stacking = False

    def set_label_opacity(self, value: int):
        self._scene.label_item.setOpacity(value / 100.0)

    def set_sam_opacity(self, value: int):
        self._scene.sam_item.setOpacity(value / 100.0)

    @pyqtSlot(bool)
    def handle_image_stacking_signal(self, is_stacking: bool):
        self.image_stacking = is_stacking

    @pyqtSlot(bool)
    def handle_sam_signal(self, is_sam: bool):
        self._sam_mode = is_sam
        self._scene.handle_sam_mode(is_sam)

    def set_brush_color(self, color: QColor):
        self._scene.set_brush_color(color)

    def set_brush_size(self, value: int):
        self._scene.set_brush_size(value)

    def set_eraser(self, value: bool):
        self._scene.set_eraser(value)

    def reset_zoom(self):
        self.fitInView(self._scene.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_label(self):
        self._scene.label_item.clear()

    def save_label_to(self, path: Path):
        self._scene.save_label(path)

    def load_sample(self, image_path: Path, label_path: Path, sam_path: Path):
        if self.image_stacking:
            # obtain path of all images with the same site
            image_stem = image_path.stem.split('I')[0]  # assuming stem is like 'site1_001'
            image_dir = image_path.parent
            print("image dir: ", image_dir)
            all_images = list(image_dir.glob(f"{image_stem}*.*"))
            if not all_images:
                raise FileNotFoundError(f"No images found for stem: {image_stem}")
            # Sort images by name to maintain order
            # convert to string paths for QPixmap compatibility
            all_images = [str(img) for img in all_images if img.is_file()]
            all_images.sort()
            print("all images: ", all_images)
            
            # create a composite image with all images stacked, with low opacity
            first_image = QPixmap(all_images[0])
            composite_image = QPixmap(first_image.size())
            composite_image.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(composite_image)
            opacity = 1.0 / len(all_images)  # Equal opacity for all images
            painter.setOpacity(opacity)
            
            for img_path in all_images:
                temp_image = QPixmap(img_path)
                if temp_image.size() != first_image.size():
                    raise ValueError(f"Image sizes do not match: {temp_image.size()} vs {first_image.size()}")
                painter.drawPixmap(0, 0, temp_image)
            painter.end()
            self._scene.setSceneRect(QRectF(QPointF(), QSizeF(composite_image.size())))
            self._scene.image_item.setPixmap(composite_image)
            
        else:
            image = QPixmap(str(image_path))
            self._scene.setSceneRect(QRectF(QPointF(), QSizeF(image.size())))
            self._scene.image_item.setPixmap(QPixmap(str(image_path)))
            if label_path.exists():
                self._scene.label_item.set_image(str(label_path))
            else:
                self._scene.label_item.clear()
            if sam_path.exists():
                self._scene.sam_item.set_image(str(sam_path))
            self.fitInView(self._scene.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self._scene.image_item)

    def scrollBy(self, point: QPoint):
        h_val = self.horizontalScrollBar().value() - point.x()
        v_val = self.verticalScrollBar().value() - point.y()
        self.horizontalScrollBar().setValue(h_val)
        self.verticalScrollBar().setValue(v_val)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_mode = True
            self._last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_mode:
            curr_pos = event.pos()
            delta = curr_pos - self._last_pos
            self.scrollBy(delta)
            self._last_pos = curr_pos
        super().mouseMoveEvent(event)  # allows proper zoom-to-cursor

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_mode = False
            self.setCursor(Qt.CursorShape.BlankCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        forward = event.angleDelta().y() > 0
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            # zoom in/out
            factor = 1.25 if forward else 0.8
            self.scale(factor, factor)
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # change brush size
            sign = -1 if forward else 1
            self._scene.change_brush_size(sign, self._brush_feedback)
