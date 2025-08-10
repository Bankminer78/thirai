import SwiftUI
import Foundation
import CoreML
import CoreImage
import AppKit
import UniformTypeIdentifiers

struct ImageRoundtripApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

ImageRoundtripApp.main()

struct ContentView: View {
    @State private var selectedImage: NSImage?
    @State private var decodedImage: NSImage?
    @State private var isProcessing = false
    @State private var encodeTime: Double = 0
    @State private var decodeTime: Double = 0
    @State private var latentStats: String = ""
    @State private var errorMessage: String = ""
    @State private var showingFileImporter = false
    
    let processor = ImageProcessor()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Image Roundtrip Encoder/Decoder")
                .font(.title)
                .padding()
            
            HStack(spacing: 40) {
                // Original Image
                VStack {
                    Text("Original Image")
                        .font(.headline)
                    
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 300, height: 300)
                        .overlay(
                            Group {
                                if let image = selectedImage {
                                    Image(nsImage: image)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                } else {
                                    VStack {
                                        Image(systemName: "photo")
                                            .font(.system(size: 50))
                                            .foregroundColor(.gray)
                                        Text("Click to upload image")
                                            .foregroundColor(.gray)
                                    }
                                }
                            }
                        )
                        .onTapGesture {
                            showingFileImporter = true
                        }
                }
                
                // Decoded Image
                VStack {
                    Text("Decoded Image")
                        .font(.headline)
                    
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 300, height: 300)
                        .overlay(
                            Group {
                                if let image = decodedImage {
                                    Image(nsImage: image)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                } else {
                                    VStack {
                                        Image(systemName: "photo.badge.arrow.down")
                                            .font(.system(size: 50))
                                            .foregroundColor(.gray)
                                        Text("Processed image will appear here")
                                            .foregroundColor(.gray)
                                    }
                                }
                            }
                        )
                }
            }
            
            // Process Button
            Button(action: processImage) {
                HStack {
                    if isProcessing {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(isProcessing ? "Processing..." : "Process Image")
                }
            }
            .disabled(selectedImage == nil || isProcessing)
            .padding()
            .background(selectedImage == nil ? Color.gray : Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
            
            // Performance Stats
            VStack(alignment: .leading, spacing: 5) {
                if encodeTime > 0 || decodeTime > 0 {
                    Text("Performance:")
                        .font(.headline)
                    HStack {
                        Text("Encode: \(String(format: "%.2f", encodeTime)) ms")
                        Spacer()
                        Text("Decode: \(String(format: "%.2f", decodeTime)) ms")
                        Spacer()
                        Text("Total: \(String(format: "%.2f", encodeTime + decodeTime)) ms")
                    }
                    .font(.monospaced(.body)())
                }
                
                if !latentStats.isEmpty {
                    Text("Latent Stats:")
                        .font(.headline)
                    Text(latentStats)
                        .font(.monospaced(.caption)())
                }
                
                if !errorMessage.isEmpty {
                    Text("Error:")
                        .font(.headline)
                        .foregroundColor(.red)
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
        }
        .padding()
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result: result)
        }
    }
    
    private func handleFileImport(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                loadImage(from: url)
            }
        case .failure(let error):
            errorMessage = "Failed to import file: \(error.localizedDescription)"
        }
    }
    
    private func loadImage(from url: URL) {
        do {
            let data = try Data(contentsOf: url)
            if let nsImage = NSImage(data: data) {
                selectedImage = nsImage
                decodedImage = nil
                encodeTime = 0
                decodeTime = 0
                latentStats = ""
                errorMessage = ""
            }
        } catch {
            errorMessage = "Failed to load image: \(error.localizedDescription)"
        }
    }
    
    private func processImage() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        errorMessage = ""
        
        Task {
            do {
                let result = try await processor.processImage(image)
                
                await MainActor.run {
                    self.decodedImage = result.decodedImage
                    self.encodeTime = result.encodeTime
                    self.decodeTime = result.decodeTime
                    self.latentStats = result.latentStats
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Processing failed: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}

struct ProcessingResult {
    let decodedImage: NSImage
    let encodeTime: Double
    let decodeTime: Double
    let latentStats: String
}

class ImageProcessor {
    private let SIDE = 384
    private var encModel: MLModel?
    private var decModel: MLModel?
    private let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    private lazy var ciCtx = CIContext(options: [.workingColorSpace: cs, .outputColorSpace: cs])
    
    init() {
        loadModels()
    }
    
    private func loadModels() {
        do {
            let basePath = "/Users/niranjanbaskaran/git/thirai"
            let encURL = URL(fileURLWithPath: "\(basePath)/build/enc/sdxl_vae_encoder_384x384.mlmodelc")
            let decURL = URL(fileURLWithPath: "\(basePath)/build/dec/sdxl_vae_decoder_384x384_img.mlmodelc")
            
            print("🔍 Loading models...")
            print("📍 Base path: \(basePath)")
            print("🔧 Encoder URL: \(encURL.path)")
            print("🔧 Decoder URL: \(decURL.path)")
            
            // Check if files exist
            let fileManager = FileManager.default
            let encExists = fileManager.fileExists(atPath: encURL.path)
            let decExists = fileManager.fileExists(atPath: decURL.path)
            
            print("📁 Encoder exists: \(encExists)")
            print("📁 Decoder exists: \(decExists)")
            
            if !encExists {
                print("❌ Encoder model not found at: \(encURL.path)")
                return
            }
            
            if !decExists {
                print("❌ Decoder model not found at: \(decURL.path)")
                return
            }
            
            let encCfg = MLModelConfiguration()
            encCfg.computeUnits = .all
            encCfg.allowLowPrecisionAccumulationOnGPU = true
            
            let decCfg = MLModelConfiguration()
            decCfg.computeUnits = .all
            decCfg.allowLowPrecisionAccumulationOnGPU = true
            
            print("⚙️  Loading encoder model...")
            encModel = try MLModel(contentsOf: encURL, configuration: encCfg)
            print("✅ Encoder model loaded successfully")
            
            print("⚙️  Loading decoder model...")
            decModel = try MLModel(contentsOf: decURL, configuration: decCfg)
            print("✅ Decoder model loaded successfully")
            
            print("🎉 All models loaded successfully!")
            
        } catch {
            print("❌ Failed to load models: \(error)")
            print("📝 Error details: \(error.localizedDescription)")
        }
    }
    
    func processImage(_ image: NSImage) async throws -> ProcessingResult {
        print("🚀 Starting image processing...")
        print("🔍 Checking model availability...")
        print("📊 Encoder model: \(encModel != nil ? "✅ Available" : "❌ Not loaded")")
        print("📊 Decoder model: \(decModel != nil ? "✅ Available" : "❌ Not loaded")")
        
        guard let encModel = encModel, let decModel = decModel else {
            print("❌ Models not loaded - cannot process image")
            throw ProcessingError.modelsNotLoaded
        }
        
        // Convert NSImage to CGImage
        var rect = CGRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            throw ProcessingError.imageConversionFailed
        }
        
        // Resize to buffer
        let pixelBuffer = try resizeToBuffer(cgImage, side: SIDE)
        
        // Warm up models
        _ = try? encModel.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(pixelBuffer: pixelBuffer)]))
        
        // Encode
        let startEnc = CFAbsoluteTimeGetCurrent()
        let encOut = try encModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(pixelBuffer: pixelBuffer)]))
        let encMS = (CFAbsoluteTimeGetCurrent() - startEnc) * 1000.0
        
        // Get latent array
        guard let zArr = getMultiArrayOutput(from: encOut) else {
            throw ProcessingError.encoderOutputNotFound
        }
        
        let stats = latentStats(zArr)
        let statsString = String(format: "shape=%@ | min=%.4f mean=%.4f std=%.4f",
                               zArr.shape.map{"\($0)"}.joined(separator: ","), stats.min, stats.mean, stats.std)
        
        // Decode
        let startDec = CFAbsoluteTimeGetCurrent()
        let decOut = try decModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["z_scaled": MLFeatureValue(multiArray: zArr)]))
        let decMS = (CFAbsoluteTimeGetCurrent() - startDec) * 1000.0
        
        // Get decoded image
        guard let decodedPixelBuffer = getImageOutput(from: decOut) else {
            throw ProcessingError.decoderOutputNotFound
        }
        
        // Convert to NSImage
        let decodedNSImage = try pixelBufferToNSImage(decodedPixelBuffer)
        
        return ProcessingResult(
            decodedImage: decodedNSImage,
            encodeTime: encMS,
            decodeTime: decMS,
            latentStats: statsString
        )
    }
    
    private func resizeToBuffer(_ cg: CGImage, side: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: side,
            kCVPixelBufferHeightKey: side,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let out = pb else { throw ProcessingError.pixelBufferCreationFailed }
        let ci = CIImage(cgImage: cg)
        let sx = CGFloat(side) / ci.extent.width
        let sy = CGFloat(side) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        ciCtx.render(scaled, to: out, bounds: CGRect(x: 0, y: 0, width: side, height: side), colorSpace: cs)
        return out
    }
    
    private func getMultiArrayOutput(from prediction: MLFeatureProvider) -> MLMultiArray? {
        for name in prediction.featureNames {
            if let value = prediction.featureValue(for: name), value.type == .multiArray {
                return value.multiArrayValue
            }
        }
        return nil
    }
    
    private func getImageOutput(from prediction: MLFeatureProvider) -> CVPixelBuffer? {
        for name in prediction.featureNames {
            if let value = prediction.featureValue(for: name), value.type == .image {
                return value.imageBufferValue
            }
        }
        return nil
    }
    
    private func latentStats(_ z: MLMultiArray) -> (min: Float, mean: Float, std: Float) {
        let n = z.count
        let ptr = z.dataPointer.bindMemory(to: Float.self, capacity: n)
        var s: Float = 0, s2: Float = 0
        var mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude
        for i in 0..<n {
            let v = ptr[i]
            s += v; s2 += v*v
            if v < mn { mn = v }
            if v > mx { mx = v }
        }
        let mean = s / Float(n)
        let var_ = max(0, s2 / Float(n) - mean * mean)
        return (mn, mean, sqrt(var_))
    }
    
    private func pixelBufferToNSImage(_ pixelBuffer: CVPixelBuffer) throws -> NSImage {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let rep = NSBitmapImageRep(ciImage: ciImage)
        let nsImage = NSImage(size: rep.size)
        nsImage.addRepresentation(rep)
        return nsImage
    }
}

enum ProcessingError: Error, LocalizedError {
    case modelsNotLoaded
    case imageConversionFailed
    case pixelBufferCreationFailed
    case encoderOutputNotFound
    case decoderOutputNotFound
    
    var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "Models not loaded. Check model paths."
        case .imageConversionFailed:
            return "Failed to convert image."
        case .pixelBufferCreationFailed:
            return "Failed to create pixel buffer."
        case .encoderOutputNotFound:
            return "Encoder output not found."
        case .decoderOutputNotFound:
            return "Decoder output not found."
        }
    }
}