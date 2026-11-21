// Run capture on the audio rendering thread, independently of page rendering.
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length || !channels[0].length) return true;
    const mono = new Float32Array(channels[0].length);
    for (const channel of channels) {
      for (let index = 0; index < mono.length; index++) {
        mono[index] += channel[index] / channels.length;
      }
    }
    this.port.postMessage(mono, [mono.buffer]);
    // Outputs remain silent: microphone audio must not feed the speakers.
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
