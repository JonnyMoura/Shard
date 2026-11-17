export function sanitizeFilename(name) {
    return String(name)
        .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
        .replace(/\s+/g, '_')
        .substring(0, 100);
}

export function audioBufferToWav(audioBuffer) {
    const numberOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1;
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;
    const samples = audioBuffer.length;
    const dataSize = samples * blockAlign;
    const bufferSize = 44 + dataSize;

    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    const channelData = [];
    for (let i = 0; i < numberOfChannels; i++) {
        channelData.push(audioBuffer.getChannelData(i));
    }

    let offset = 44;
    for (let i = 0; i < samples; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, int16, true);
            offset += 2;
        }
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

export function mixAudioBuffers(buffers, audioContext) {
    if (buffers.length === 0) return null;
    if (buffers.length === 1) return buffers[0];

    const maxLength = Math.max(...buffers.map(buffer => buffer.length));
    const numberOfChannels = Math.max(...buffers.map(buffer => buffer.numberOfChannels));
    const sampleRate = buffers[0].sampleRate;
    const mixedBuffer = audioContext.createBuffer(
        numberOfChannels,
        maxLength,
        sampleRate
    );

    for (let channel = 0; channel < numberOfChannels; channel++) {
        const outputData = mixedBuffer.getChannelData(channel);

        for (const buffer of buffers) {
            const channelIndex = Math.min(channel, buffer.numberOfChannels - 1);
            const inputData = buffer.getChannelData(channelIndex);

            for (let i = 0; i < inputData.length; i++) {
                outputData[i] = (outputData[i] || 0) + inputData[i] / buffers.length;
            }
        }

        let max = 0;
        for (let i = 0; i < outputData.length; i++) {
            const abs = Math.abs(outputData[i]);
            if (abs > max) max = abs;
        }
        if (max > 1) {
            for (let i = 0; i < outputData.length; i++) {
                outputData[i] /= max;
            }
        }
    }

    return mixedBuffer;
}
