
import Foundation
import CoreML
import CoreImage
import AppKit
import Accelerate

// -------- logging --------
@inline(__always)
func nowTS() -> String {
    let df = DateFormatter()
    df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
    return df.string(from: Date())
}
@inline(__always)
func log(_ msg: String) { print("[\(nowTS())] \(msg)") }

// -------- config --------
let SIDE = 256
let ENC_URL = URL(fileURLWithPath: "build/enc/sdxl_vae_encoder_\(SIDE)x\(SIDE).mlmodelc")
let DEC_URL = URL(fileURLWithPath: "build/dec/sdxl_vae_decoder_\(SIDE)x\(SIDE).mlmodelc")

// Environment + paths
log("CWD: \(FileManager.default.currentDirectoryPath)")
log("Encoder path: \(ENC_URL.path)")
log("Decoder path: \(DEC_URL.path)")

// Validate model paths early for clear errors
let fm = FileManager.default
guard fm.fileExists(atPath: ENC_URL.path) else { fatalError("Encoder model not found at: \(ENC_URL.path)") }
guard fm.fileExists(atPath: DEC_URL.path) else { fatalError("Decoder model not found at: \(DEC_URL.path)") }

// -------- utils --------
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ciCtx = CIContext(options: [.workingColorSpace: cs, .outputColorSpace: cs])

func loadCGImage(_ path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    guard let img = NSImage(data: data) else { throw NSError(domain: "img", code: -1) }
    var rect = CGRect(origin: .zero, size: img.size)
    guard let cg = img.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
        throw NSError(domain: "img", code: -2)
    }
    return cg
}

func resizeToBuffer(_ cg: CGImage, side: Int) throws -> CVPixelBuffer {
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
    guard let out = pb else { throw NSError(domain: "pb", code: -1) }
    let ci = CIImage(cgImage: cg)
    let sx = CGFloat(side) / ci.extent.width
    let sy = CGFloat(side) / ci.extent.height
    let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
    ciCtx.render(scaled, to: out, bounds: CGRect(x: 0, y: 0, width: side, height: side), colorSpace: cs)
    return out
}

// Convert BGRA CVPixelBuffer (0..255) -> MLMultiArray [1,3,H,W] in [-1,1]
func bufferToNCHWMinusOneToOne(_ pb: CVPixelBuffer) throws -> MLMultiArray {
    let H = CVPixelBufferGetHeight(pb)
    let W = CVPixelBufferGetWidth(pb)
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: H), NSNumber(value: W)], dataType: .float32)
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    var idx = 0
    for y in 0..<H {
        let row = base + y*bytesPerRow
        for x in 0..<W {
            let b = Float(row[x*4 + 0])
            let g = Float(row[x*4 + 1])
            let r = Float(row[x*4 + 2])
            let rf = (r / 127.5) - 1.0
            let gf = (g / 127.5) - 1.0
            let bf = (b / 127.5) - 1.0
            let baseR = 0
            let baseG = H*W
            let baseB = 2*H*W
            arr[baseR + idx] = NSNumber(value: rf)
            arr[baseG + idx] = NSNumber(value: gf)
            arr[baseB + idx] = NSNumber(value: bf)
            idx += 1
        }
    }
    return arr
}

func savePixelBufferPNG(_ pb: CVPixelBuffer, to path: String) throws {
    let ci = CIImage(cvPixelBuffer: pb)
    let rep = NSBitmapImageRep(ciImage: ci)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "png", code: -1)
    }
    try data.write(to: URL(fileURLWithPath: path))
}

func saveLatentBIN(_ z: MLMultiArray, to path: String) throws {
    let ptr = z.dataPointer.bindMemory(to: Float.self, capacity: z.count)
    let buf = UnsafeBufferPointer(start: ptr, count: z.count)
    let data = Data(buffer: buf)
    try data.write(to: URL(fileURLWithPath: path))
}

@inline(__always)
func latentStats(_ z: MLMultiArray) -> (min: Float, mean: Float, std: Float) {
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

func dumpModelIO(_ model: MLModel, name: String) {
    let md = model.modelDescription
    log("Model IO for \(name):")
    for (k, d) in md.inputDescriptionsByName { log("  input[\(k)]: type=\(d.type)") }
    for (k, d) in md.outputDescriptionsByName { log("  output[\(k)]: type=\(d.type)") }
}

// -------- main --------
guard CommandLine.arguments.count >= 2 else {
    print("Usage: runner <input_image_path>"); exit(1)
}
let inPath = CommandLine.arguments[1]
log("Input image: \(inPath)")

// Model configurations: prefer GPU, allow low-precision accumulation (safe for VAE)
let encCfg = MLModelConfiguration()
encCfg.computeUnits = .all
encCfg.allowLowPrecisionAccumulationOnGPU = true

let decCfg = MLModelConfiguration()
decCfg.computeUnits = .all
decCfg.allowLowPrecisionAccumulationOnGPU = true

var encModel: MLModel
var decModel: MLModel

do {
    log("Loading encoder…")
    encModel = try MLModel(contentsOf: ENC_URL, configuration: encCfg)
    dumpModelIO(encModel, name: "encoder")
} catch {
    log("ERROR loading encoder: \(error)")
    exit(2)
}

do {
    log("Loading decoder…")
    decModel = try MLModel(contentsOf: DEC_URL, configuration: decCfg)
    dumpModelIO(decModel, name: "decoder")
} catch {
    log("ERROR loading decoder: \(error)")
    exit(3)
}

let cg = try loadCGImage(inPath)
let inPB = try resizeToBuffer(cg, side: SIDE)

// Warm up encoder once with appropriate input type
let encInputType = encModel.modelDescription.inputDescriptionsByName["x"]?.type ?? .invalid
log("Encoder input type: \(encInputType)")
if encInputType == .image {
    _ = try? encModel.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(pixelBuffer: inPB)]))
} else if encInputType == .multiArray {
    if let xArr = try? bufferToNCHWMinusOneToOne(inPB) {
        _ = try? encModel.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: xArr)]))
    }
}

// ---- ENCODE ---- handle both input kinds
let startEnc = CFAbsoluteTimeGetCurrent()
let encOut: MLFeatureProvider
if encInputType == .image {
    log("Encode via ImageType input")
    encOut = try encModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(pixelBuffer: inPB)]))
} else if encInputType == .multiArray {
    log("Encode via MultiArray input [-1,1] NCHW")
    let xArr = try bufferToNCHWMinusOneToOne(inPB)
    encOut = try encModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: xArr)]))
} else {
    fatalError("Unsupported encoder input type: \(encInputType)")
}
let encMS = (CFAbsoluteTimeGetCurrent() - startEnc) * 1000.0
log(String(format: "Encode completed in %.2f ms", encMS))

log("Encoder outputs:")
for name in encOut.featureNames {
    if let v = encOut.featureValue(for: name) {
        log("  name=\(name) type=\(v.type)")
    }
}

// Grab the MLMultiArray output as z_scaled
var zArr: MLMultiArray?
for name in encOut.featureNames {
    if let v = encOut.featureValue(for: name), v.type == .multiArray {
        zArr = v.multiArrayValue
        break
    }
}
guard let z = zArr else { fatalError("Encoder output MLMultiArray not found") }

let stats = latentStats(z)
log(String(format: "z_scaled shape=%@ | min=%.4f mean=%.4f std=%.4f",
           z.shape.map{"\($0)"}.joined(separator: ","), stats.min, stats.mean, stats.std))
try saveLatentBIN(z, to: "latent.bin")

// ---- DECODE ---- decoder expects z_scaled and outputs ImageType (CVPixelBuffer)
let startDec = CFAbsoluteTimeGetCurrent()
let decOut = try decModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["z_scaled": MLFeatureValue(multiArray: z)]))
let decMS = (CFAbsoluteTimeGetCurrent() - startDec) * 1000.0
log(String(format: "Decode completed in %.2f ms", decMS))

log("Decoder outputs:")
for name in decOut.featureNames {
    if let v = decOut.featureValue(for: name) {
        log("  name=\(name) type=\(v.type)")
    }
}

var outPB: CVPixelBuffer?
for name in decOut.featureNames {
    if let v = decOut.featureValue(for: name), v.type == .image {
        outPB = v.imageBufferValue
        break
    }
}

guard let decoded = outPB else { fatalError("Decoder image output not found") }
try savePixelBufferPNG(decoded, to: "out.png")

print(String(format: "OK | encode: %.2f ms  decode: %.2f ms  (saved out.png, latent.bin)", encMS, decMS))