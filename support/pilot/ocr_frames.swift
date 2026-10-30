import Foundation
import AVFoundation
import Vision
import CryptoKit
import CoreVideo

struct OCRFailure: Error, CustomStringConvertible {
    let description: String
}

@main
struct FullFrameOCR {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 4 else {
            throw OCRFailure(description: "usage: ocr_frames VIDEO EXPECTED_JSON OUTPUT_JSONL")
        }
        let input = URL(fileURLWithPath: args[1])
        let expectedData = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        guard let expected = try JSONSerialization.jsonObject(with: expectedData) as? [[String: Any]] else {
            throw OCRFailure(description: "invalid expected frame ledger")
        }
        let destination = URL(fileURLWithPath: args[3])
        let temporary = destination.appendingPathExtension("tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        let handle = try FileHandle(forWritingTo: temporary)
        defer { try? handle.close() }
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw OCRFailure(description: "no video track")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw OCRFailure(description: "reader cannot add video") }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? OCRFailure(description: "reader start failed")
        }
        var index = 0
        var previousHash = ""
        var previousTexts = [[String: Any]]()
        var previousID = ""
        let start = Date()
        while let sample = output.copyNextSampleBuffer() {
            try autoreleasepool {
                guard index < expected.count else {
                    throw OCRFailure(description: "native decoder emitted extra frame")
                }
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else {
                    throw OCRFailure(description: "missing image buffer")
                }
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let actualTime = CMTimeGetSeconds(pts)
                let reference = expected[index]
                let wantedTime = reference["part_time"] as! Double
                guard actualTime.isFinite, abs(actualTime - wantedTime) < 0.000064 else {
                    throw OCRFailure(description: "native/source timestamp mismatch at \(index)")
                }
                CVPixelBufferLockBaseAddress(buffer, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
                let stride = CVPixelBufferGetBytesPerRow(buffer)
                let height = CVPixelBufferGetHeight(buffer)
                guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                    throw OCRFailure(description: "image buffer has no base address")
                }
                let pixels = Data(bytesNoCopy: base, count: stride * height, deallocator: .none)
                let hash = SHA256.hash(data: pixels).map { String(format: "%02x", $0) }.joined()
                let frameID = reference["frame_id"] as! String
                var texts = [[String: Any]]()
                var reused: String? = nil
                if hash == previousHash {
                    texts = previousTexts
                    reused = previousID
                } else {
                    let request = VNRecognizeTextRequest()
                    request.recognitionLevel = .accurate
                    request.recognitionLanguages = ["zh-Hans", "en-US"]
                    request.usesLanguageCorrection = false
                    request.minimumTextHeight = 0.005
                    try VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
                    texts = (request.results ?? []).compactMap { observation -> [String: Any]? in
                        guard let best = observation.topCandidates(1).first else { return nil }
                        let b = observation.boundingBox
                        return ["text": best.string, "confidence": best.confidence,
                                "bbox_bottom_left": [b.origin.x, b.origin.y, b.width, b.height]]
                    }
                }
                var record: [String: Any] = [
                    "schema_version": "0.1", "frame_id": frameID, "index": index,
                    "native_pts": pts.value, "native_time_base": "1/\(pts.timescale)",
                    "part_time": actualTime, "source_pts_matched": true,
                    "width": CVPixelBufferGetWidth(buffer), "height": height,
                    "pixel_format": "BGRA", "bytes_per_row": stride,
                    "pixel_sha256_including_stride": hash,
                    "ocr_engine": "AppleVision", "language_correction": false,
                    "status": "success", "texts": texts
                ]
                if let reused = reused { record["reused_exact_pixels_from"] = reused }
                var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
                data.append(0x0A)
                try handle.write(contentsOf: data)
                previousHash = hash
                previousTexts = texts
                previousID = frameID
                index += 1
                if index % 30 == 0 || index == expected.count {
                    let message = "OCR \(index)/\(expected.count), seconds \(String(format: "%.1f", Date().timeIntervalSince(start)))\n"
                    FileHandle.standardError.write(message.data(using: .utf8)!)
                }
            }
        }
        guard reader.status == .completed else {
            throw reader.error ?? OCRFailure(description: "native reader incomplete")
        }
        guard index == expected.count else {
            throw OCRFailure(description: "native decoder ended early \(index)/\(expected.count)")
        }
        try handle.synchronize()
        if FileManager.default.fileExists(atPath: destination.path) {
            throw OCRFailure(description: "destination already exists; preserve previous result")
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
        let summary: [String: Any] = ["frames": index, "elapsed": Date().timeIntervalSince(start),
                                     "all_pts_matched": true, "output": destination.path]
        print(String(data: try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]), encoding: .utf8)!)
    }
}
