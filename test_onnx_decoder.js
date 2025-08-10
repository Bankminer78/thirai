#!/usr/bin/env node

const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function decodeLatent(latentPath, modelPath, outputDir = './output') {
    try {
        console.log(`Loading model: ${modelPath}`);
        const session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'], // Node.js typically uses CPU
            graphOptimizationLevel: 'all'
        });
        console.log('Model loaded successfully');

        // Read latent file
        console.log(`Reading latent file: ${latentPath}`);
        const latentBuffer = fs.readFileSync(latentPath);
        
        // Assume fp16 format (Uint16Array)
        const u16Array = new Uint16Array(latentBuffer.buffer, latentBuffer.byteOffset, latentBuffer.length / 2);
        console.log(`Latent elements: ${u16Array.length}`);
        
        // Calculate dimensions (assuming square latent with 4 channels)
        const spatialElements = u16Array.length / 4;
        const latentSize = Math.sqrt(spatialElements);
        if (Math.floor(latentSize) !== latentSize) {
            throw new Error(`Invalid latent dimensions. Elements: ${u16Array.length}, spatial: ${spatialElements}`);
        }
        
        console.log(`Latent shape: [1, 4, ${latentSize}, ${latentSize}]`);
        
        // Create tensor (convert fp16 to fp32 for compatibility)
        const f32Array = new Float32Array(u16Array.length);
        for (let i = 0; i < u16Array.length; i++) {
            // Simple fp16 to fp32 conversion (this is approximate)
            f32Array[i] = u16Array[i] / 65535.0 * 2.0 - 1.0; // normalize to [-1,1]
        }
        
        const inputTensor = new ort.Tensor('float32', f32Array, [1, 4, latentSize, latentSize]);
        
        // Run inference
        console.log('Running inference...');
        const start = performance.now();
        const results = await session.run({ z_scaled: inputTensor });
        const end = performance.now();
        console.log(`Inference completed in ${(end - start).toFixed(2)}ms`);
        
        // Get output tensor
        const outputTensor = results.y;
        console.log(`Output shape: [${outputTensor.dims.join(', ')}]`);
        console.log(`Output type: ${outputTensor.type}`);
        
        if (outputTensor.dims.length !== 4 || outputTensor.dims[1] !== 3) {
            throw new Error(`Unexpected output shape: [${outputTensor.dims.join(', ')}]`);
        }
        
        const [batch, channels, height, width] = outputTensor.dims;
        const outputData = outputTensor.data;
        
        // Convert to image data (RGB, 0-255)
        const imageData = new Uint8Array(height * width * 3);
        const planeSize = height * width;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const imgIdx = idx * 3;
                
                // Get RGB values from separate planes and convert from [-1,1] to [0,255]
                const r = Math.max(0, Math.min(255, Math.round((outputData[idx] + 1) * 127.5)));
                const g = Math.max(0, Math.min(255, Math.round((outputData[idx + planeSize] + 1) * 127.5)));
                const b = Math.max(0, Math.min(255, Math.round((outputData[idx + 2 * planeSize] + 1) * 127.5)));
                
                imageData[imgIdx] = r;
                imageData[imgIdx + 1] = g;
                imageData[imgIdx + 2] = b;
            }
        }
        
        // Save as PNG using Sharp
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        const pngPath = path.join(outputDir, 'decoded_image.png');
        await sharp(Buffer.from(imageData), {
            raw: {
                width: width,
                height: height,
                channels: 3
            }
        })
        .png()
        .toFile(pngPath);
        
        console.log(`PNG image saved to: ${pngPath}`);
        console.log(`Image dimensions: ${width}x${height}`);
        
        // Also save as JPG
        const jpgPath = path.join(outputDir, 'decoded_image.jpg');
        await sharp(Buffer.from(imageData), {
            raw: {
                width: width,
                height: height,
                channels: 3
            }
        })
        .jpeg({ quality: 90 })
        .toFile(jpgPath);
        
        console.log(`JPG image saved to: ${jpgPath}`);
        
        return pngPath;
        
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

// Main execution
if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.log('Usage: node test_onnx_decoder.js <latent_file> <model_file> [output_dir]');
        console.log('Example: node test_onnx_decoder.js latent_fp16.bin sdxl_vae_decoder_fp32.onnx ./output');
        process.exit(1);
    }
    
    const [latentPath, modelPath, outputDir] = args;
    decodeLatent(latentPath, modelPath, outputDir);
}

module.exports = { decodeLatent };