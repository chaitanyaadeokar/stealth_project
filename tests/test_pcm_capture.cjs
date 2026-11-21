const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('worklet preserves input samples and averages multiple channels', () => {
  const frames = [];
  let Processor;
  const source = fs.readFileSync(path.join(__dirname,
    '../backend/app/static/js/pcm-capture.js'), 'utf8');
  vm.runInNewContext(source, {
    AudioWorkletProcessor: class {
      constructor() { this.port = { postMessage: (frame) => frames.push(frame) }; }
    },
    registerProcessor: (name, implementation) => { Processor = implementation; },
    Float32Array,
  });
  const capture = new Processor();
  const left = new Float32Array([0.25, -0.5, 0.75]);
  assert.equal(capture.process([[left]]), true);
  assert.deepEqual(Array.from(frames[0]), Array.from(left));
  capture.process([[left, new Float32Array([-0.25, 0.5, 0.25])]]);
  assert.deepEqual(Array.from(frames[1]), [0, 0, 0.5]);
  capture.process([[]]);
  assert.equal(frames.length, 2);
});
