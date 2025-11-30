#!/usr/bin/env node
/**
 * Convert JSONL OCR output to Toon format
 * Usage: node convert_to_toon.mjs <input.jsonl> <output.toon>
 */

import { readFile, writeFile } from 'fs/promises';
import { encode } from '@toon-format/toon';

async function convertToToon(inputPath, outputPath) {
  try {
    console.log(`📖 Reading input file: ${inputPath}`);
    const content = await readFile(inputPath, 'utf-8');

    // Parse JSONL
    const lines = content.trim().split('\n');
    const pages = [];

    for (const line of lines) {
      if (!line.trim()) continue;

      try {
        const entry = JSON.parse(line);
        pages.push(entry);
      } catch (err) {
        console.warn(`⚠️  Skipping invalid JSON line: ${err.message}`);
      }
    }

    console.log(`✅ Parsed ${pages.length} pages`);

    // Create Toon-compatible structure
    const toonData = {
      metadata: {
        source: 'ai-drawing-analyzer',
        version: '2.2.0',
        pages: pages.length,
        provider: pages[0]?.provider || 'unknown',
        model: pages[0]?.model || 'unknown',
        timestamp: new Date().toISOString()
      },
      content: pages.map(page => ({
        page: page.page,
        type: page.page_type || 'image',
        text: page.text_content || '',
        timestamp: page.timestamp
      }))
    };

    console.log(`🔄 Encoding to Toon format...`);
    const encoded = encode(toonData);

    console.log(`💾 Writing output file: ${outputPath}`);
    await writeFile(outputPath, encoded);

    console.log(`✅ Successfully converted to Toon format!`);
    console.log(`   Input:  ${inputPath}`);
    console.log(`   Output: ${outputPath}`);
    console.log(`   Pages:  ${pages.length}`);

    return 0;
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    console.error(error.stack);
    return 1;
  }
}

// Main execution
const args = process.argv.slice(2);

if (args.length < 2) {
  console.error('Usage: node convert_to_toon.mjs <input.jsonl> <output.toon>');
  process.exit(1);
}

const [inputPath, outputPath] = args;
convertToToon(inputPath, outputPath).then(code => process.exit(code));
