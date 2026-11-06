import SwiftUI
import UIKit

/// The system camera, wrapped for the composer's attach menu. Presents
/// `UIImagePickerController` in camera mode and hands the shot back — the
/// only camera surface in the app, so the app target carries the camera
/// usage string for it (`INFOPLIST_KEY_NSCameraUsageDescription`).
struct CameraPicker: UIViewControllerRepresentable {
    let onShot: (UIImage) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onShot: onShot)
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onShot: (UIImage) -> Void

        init(onShot: @escaping (UIImage) -> Void) {
            self.onShot = onShot
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                onShot(image)
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
