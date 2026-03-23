import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import PyNearDemo 1.0

ApplicationWindow {
    id: root
    title: "PyNear · KNN Demo"
    width: 1100
    height: 720
    minimumWidth: 700
    minimumHeight: 480
    visible: true
    color: "#0d0d1a"

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── Controls panel ────────────────────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 210
            Layout.fillHeight: true
            color: "#12121e"

            ColumnLayout {
                anchors { fill: parent; margins: 20 }
                spacing: 14

                Label {
                    text: "PyNear"
                    font.pixelSize: 20
                    font.bold: true
                    color: "#a0a0ff"
                }
                Label {
                    text: "KNN Demo"
                    font.pixelSize: 13
                    color: "#606080"
                }

                Rectangle { height: 1; color: "#22223a"; Layout.fillWidth: true }

                // ── Points ────────────────────────────────────────────────────
                Label { text: "Points"; color: "#9090b0"; font.pixelSize: 11 }
                RowLayout {
                    Layout.fillWidth: true
                    Slider {
                        id: pointsSlider
                        Layout.fillWidth: true
                        from: 0; to: 6; stepSize: 1; value: 3
                        property var steps: [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
                        property int pointCount: steps[Math.round(value)]
                    }
                    Label {
                        text: {
                            var n = pointsSlider.pointCount
                            return n >= 1000000 ? "1M" : n >= 1000 ? (n/1000)+"K" : n
                        }
                        color: "#c0c0e0"; font.pixelSize: 12
                        Layout.preferredWidth: 36
                        horizontalAlignment: Text.AlignRight
                    }
                }

                // ── k neighbors ───────────────────────────────────────────────
                Label { text: "k neighbors"; color: "#9090b0"; font.pixelSize: 11 }
                RowLayout {
                    Layout.fillWidth: true
                    Slider {
                        id: kSlider
                        Layout.fillWidth: true
                        from: 1; to: 50; stepSize: 1; value: 10
                    }
                    Label {
                        text: Math.round(kSlider.value)
                        color: "#c0c0e0"; font.pixelSize: 12
                        Layout.preferredWidth: 28
                        horizontalAlignment: Text.AlignRight
                    }
                }

                // ── Point size ────────────────────────────────────────────────
                Label { text: "Point size"; color: "#9090b0"; font.pixelSize: 11 }
                RowLayout {
                    Layout.fillWidth: true
                    Slider {
                        id: sizeSlider
                        Layout.fillWidth: true
                        from: 1; to: 8; stepSize: 1; value: 2
                        onValueChanged: view.setPointSize(Math.round(value))
                    }
                    Label {
                        text: Math.round(sizeSlider.value) + "px"
                        color: "#c0c0e0"; font.pixelSize: 12
                        Layout.preferredWidth: 28
                        horizontalAlignment: Text.AlignRight
                    }
                }

                // ── Buttons ───────────────────────────────────────────────────
                Button {
                    id: regenButton
                    text: "Regenerate"
                    Layout.fillWidth: true
                    onClicked: view.regenerate(pointsSlider.pointCount, Math.round(kSlider.value))
                    background: Rectangle {
                        color: regenButton.pressed ? "#3a3aff"
                             : regenButton.hovered  ? "#2a2aee" : "#1e1ecc"
                        radius: 6
                    }
                    contentItem: Text {
                        text: regenButton.text; color: "white"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }

                Button {
                    id: resetButton
                    text: "Reset view"
                    Layout.fillWidth: true
                    onClicked: view.resetView()
                    background: Rectangle {
                        color: resetButton.pressed ? "#2a3a2a"
                             : resetButton.hovered  ? "#1e2e1e" : "#162216"
                        radius: 6
                        border.color: "#2a4a2a"; border.width: 1
                    }
                    contentItem: Text {
                        text: resetButton.text; color: "#80c080"
                        font.pixelSize: 12
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }

                Rectangle { height: 1; color: "#22223a"; Layout.fillWidth: true }

                // ── Status ────────────────────────────────────────────────────
                Label {
                    text: view.status
                    color: "#7070a0"; wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    font.pixelSize: 11; lineHeight: 1.4
                }

                Item { Layout.fillHeight: true }

                // ── Legend ────────────────────────────────────────────────────
                Label { text: "Neighbor rank"; color: "#505070"; font.pixelSize: 10 }
                RowLayout {
                    spacing: 4
                    Rectangle { width: 10; height: 10; radius: 5; color: "#ff4000" }
                    Label { text: "closest"; color: "#606080"; font.pixelSize: 10 }
                    Item { Layout.fillWidth: true }
                    Rectangle { width: 10; height: 10; radius: 5; color: "#603c3c" }
                    Label { text: "farthest"; color: "#606080"; font.pixelSize: 10 }
                }

                Rectangle { height: 1; color: "#22223a"; Layout.fillWidth: true }

                // ── Controls hint ─────────────────────────────────────────────
                Label {
                    text: "🖱 hover → search\n🖱 drag → pan\n🖱 scroll → zoom"
                    color: "#404060"; font.pixelSize: 10; lineHeight: 1.5
                    Layout.fillWidth: true
                }
            }
        }

        // ── Point cloud ───────────────────────────────────────────────────────
        PointCloudView {
            id: view
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Track drag start for pan
            property real dragStartX: 0
            property real dragStartY: 0

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true

                onPressed: mouse => {
                    view.dragStartX = mouse.x
                    view.dragStartY = mouse.y
                }

                onPositionChanged: mouse => {
                    if (mouse.buttons & Qt.LeftButton) {
                        view.panBy(mouse.x - view.dragStartX, mouse.y - view.dragStartY)
                        view.dragStartX = mouse.x
                        view.dragStartY = mouse.y
                    } else {
                        view.hoverAt(mouse.x, mouse.y, Math.round(kSlider.value))
                    }
                }

                onWheel: wheel => view.zoomAt(wheel.angleDelta.y, wheel.x, wheel.y)
            }
        }
    }
}
