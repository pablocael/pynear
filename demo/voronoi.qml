import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import PyNearDemo 1.0

ApplicationWindow {
    id: root
    title: "PyNear · Voronoi Demo"
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
                    text: "Voronoi Demo"
                    font.pixelSize: 13
                    color: "#606080"
                }

                Rectangle { height: 1; color: "#22223a"; Layout.fillWidth: true }

                // ── Random seed count ─────────────────────────────────────────
                Label { text: "Seed count"; color: "#9090b0"; font.pixelSize: 11 }
                RowLayout {
                    Layout.fillWidth: true
                    Slider {
                        id: seedCountSlider
                        Layout.fillWidth: true
                        from: 2; to: 64; stepSize: 1; value: 12
                    }
                    Label {
                        text: Math.round(seedCountSlider.value)
                        color: "#c0c0e0"; font.pixelSize: 12
                        Layout.preferredWidth: 28
                        horizontalAlignment: Text.AlignRight
                    }
                }

                Button {
                    id: randomizeButton
                    text: "Randomize"
                    Layout.fillWidth: true
                    onClicked: view.randomize(Math.round(seedCountSlider.value))
                    background: Rectangle {
                        color: randomizeButton.pressed ? "#3a3aff"
                             : randomizeButton.hovered  ? "#2a2aee" : "#1e1ecc"
                        radius: 6
                    }
                    contentItem: Text {
                        text: randomizeButton.text; color: "white"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }

                Button {
                    id: clearButton
                    text: "Clear"
                    Layout.fillWidth: true
                    onClicked: view.clearSeeds()
                    background: Rectangle {
                        color: clearButton.pressed ? "#3a1a1a"
                             : clearButton.hovered  ? "#2e1212" : "#220e0e"
                        radius: 6
                        border.color: "#4a2222"; border.width: 1
                    }
                    contentItem: Text {
                        text: clearButton.text; color: "#c06060"
                        font.pixelSize: 13
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

                Rectangle { height: 1; color: "#22223a"; Layout.fillWidth: true }

                // ── Live seed list ────────────────────────────────────────────
                Label { text: "Seeds  (" + view.seedCount + ")"; color: "#9090b0"; font.pixelSize: 11 }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 120
                    clip: true

                    Column {
                        spacing: 3
                        width: parent.width

                        Repeater {
                            model: view.seedCount
                            delegate: Row {
                                spacing: 6
                                Rectangle {
                                    width: 10; height: 10; radius: 5
                                    y: 2
                                    color: Qt.hsla(
                                        ((index * 137.508) % 360) / 360,
                                        0.50, 0.65, 1.0
                                    )
                                }
                                Label {
                                    text: "Seed " + (index + 1)
                                    color: "#8080aa"; font.pixelSize: 10
                                }
                            }
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                // ── Hint ──────────────────────────────────────────────────────
                Label {
                    text: "🖱 click → add seed\n🖱 drag → move seed\n🖱 right-click → remove"
                    color: "#404060"; font.pixelSize: 10; lineHeight: 1.5
                    Layout.fillWidth: true
                }
            }
        }

        // ── Voronoi canvas ────────────────────────────────────────────────────
        VoronoiView {
            id: view
            Layout.fillWidth: true
            Layout.fillHeight: true

            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.LeftButton | Qt.RightButton

                onPressed: mouse => {
                    view.mousePressed(mouse.x, mouse.y, mouse.button === Qt.RightButton)
                }
                onPositionChanged: mouse => {
                    if (mouse.buttons & Qt.LeftButton)
                        view.mouseDragged(mouse.x, mouse.y)
                }
                onReleased: view.mouseReleased()
            }
        }
    }
}
