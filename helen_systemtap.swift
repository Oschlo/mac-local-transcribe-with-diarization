// helen_systemtap.swift — fanger systemlyd ikke-destruktivt via Core Audio process tap
// (macOS 14.2+). Brukes som motpart-kanal i Helen-opptak: motpartens stemme spiller
// normalt i AirPods/headset, og vi tapper signalet uten aa omdirigere output.
//
// Bruk:  helen_systemtap <ut.wav>     # tar opp til SIGINT (Ctrl-C / kill -INT)
//
// Skriver mono 16-bit PCM WAV ved tappens native samplerate.
// Bygg:  swiftc -O helen_systemtap.swift -o helen_systemtap

import Foundation
import CoreAudio
import AudioToolbox

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("bruk: helen_systemtap <ut.wav> | --stream\n".data(using: .utf8)!)
    exit(2)
}
// --stream: emit raw s16le mono til stdout live (for Soniox realtime).
// <ut.wav>: akkumuler og skriv WAV ved stopp (post-call).
let streaming = (args[1] == "--stream")
let outPath = streaming ? "" : args[1]
let stdoutHandle = FileHandle.standardOutput

// ---- 1. Lag process tap som fanger global systemlyd (mono), uten aa mute output ----
let tapDesc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
tapDesc.isPrivate = true
tapDesc.muteBehavior = .unmuted   // motparten hoeres fortsatt i AirPods

var tapID = AudioObjectID(kAudioObjectUnknown)
var err = AudioHardwareCreateProcessTap(tapDesc, &tapID)
guard err == noErr, tapID != kAudioObjectUnknown else {
    FileHandle.standardError.write("AudioHardwareCreateProcessTap feilet: \(err)\n".data(using: .utf8)!)
    exit(1)
}

// ---- 2. Hent tappens stream-format ----
var fmt = AudioStreamBasicDescription()
var sz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var addr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
err = AudioObjectGetPropertyData(tapID, &addr, 0, nil, &sz, &fmt)
guard err == noErr else {
    FileHandle.standardError.write("kunne ikke hente tap-format: \(err)\n".data(using: .utf8)!)
    exit(1)
}
let sampleRate = fmt.mSampleRate
let inChannels = Int(fmt.mChannelsPerFrame)

// ---- 3. Lag aggregat-enhet som inneholder tappen ----
// Unik UID per prosess: en drept tap (ikke ren SIGINT) etterlater ellers en
// spoekelse-aggregatenhet med samme UID -> neste tap kolliderer = "No audio".
let aggUID = "ai.helen.systemtap.agg.\(getpid())"
let aggDesc: [String: Any] = [
    kAudioAggregateDeviceNameKey as String: "Helen System Tap",
    kAudioAggregateDeviceUIDKey as String: aggUID,
    kAudioAggregateDeviceIsPrivateKey as String: true,
    kAudioAggregateDeviceIsStackedKey as String: false,
    kAudioAggregateDeviceTapListKey as String: [
        [ kAudioSubTapUIDKey as String: tapDesc.uuid.uuidString,
          kAudioSubTapDriftCompensationKey as String: true ]
    ],
]
var aggID = AudioObjectID(kAudioObjectUnknown)
err = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
guard err == noErr, aggID != kAudioObjectUnknown else {
    FileHandle.standardError.write("AudioHardwareCreateAggregateDevice feilet: \(err)\n".data(using: .utf8)!)
    AudioHardwareDestroyProcessTap(tapID)
    exit(1)
}

// ---- WAV-skriving: akkumuler int16-mono i minne, skriv header + data ved stopp ----
var pcm = Data()
pcm.reserveCapacity(1 << 22)
var totalFrames = 0
var callbackCount = 0

func appendSamples(_ ptr: UnsafePointer<Float>, frames: Int, channels: Int) {
    // mikser ned til mono og konverterer float->int16
    var out = Data(capacity: frames * 2)
    var i = 0
    while i < frames {
        var acc: Float = 0
        for c in 0..<channels { acc += ptr[i * channels + c] }
        var s = acc / Float(channels)
        if s > 1 { s = 1 }; if s < -1 { s = -1 }
        var v = Int16(s * 32767.0)
        withUnsafeBytes(of: &v) { out.append(contentsOf: $0) }
        i += 1
    }
    if streaming {
        stdoutHandle.write(out)          // live PCM -> Soniox
    } else {
        pcm.append(out)                  // akkumuler for WAV
    }
    totalFrames += frames
}

// ---- 4. IOProc som mottar tappet lyd ----
var ioProcID: AudioDeviceIOProcID?
let ioProc: AudioDeviceIOProc = { (_, _, inInputData, _, _, _, _) -> OSStatus in
    callbackCount += 1
    let ablPtr = UnsafeMutableAudioBufferListPointer(
        UnsafeMutablePointer(mutating: inInputData))
    for b in ablPtr {
        guard let mData = b.mData else { continue }
        let ch = Int(b.mNumberChannels)
        if ch == 0 { continue }
        let frames = Int(b.mDataByteSize) / (MemoryLayout<Float>.size * ch)
        appendSamples(mData.assumingMemoryBound(to: Float.self), frames: frames, channels: ch)
    }
    return noErr
}

err = AudioDeviceCreateIOProcID(aggID, ioProc, nil, &ioProcID)
guard err == noErr, let proc = ioProcID else {
    FileHandle.standardError.write("AudioDeviceCreateIOProcID feilet: \(err)\n".data(using: .utf8)!)
    AudioHardwareDestroyAggregateDevice(aggID); AudioHardwareDestroyProcessTap(tapID)
    exit(1)
}

func cleanupAndWrite() {
    AudioDeviceStop(aggID, proc)
    AudioDeviceDestroyIOProcID(aggID, proc)
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
    if streaming {
        try? stdoutHandle.synchronize()
        FileHandle.standardError.write("stream slutt (\(totalFrames) frames)\n".data(using: .utf8)!)
        return
    }
    // ---- WAV-header (PCM16 mono) ----
    let dataLen = pcm.count
    let sr = UInt32(sampleRate)
    let byteRate = sr * 2
    var h = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { h.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { h.append(contentsOf: $0) } }
    h.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + dataLen))
    h.append(contentsOf: Array("WAVE".utf8))
    h.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
    u32(sr); u32(byteRate); u16(2); u16(16)
    h.append(contentsOf: Array("data".utf8)); u32(UInt32(dataLen))
    h.append(pcm)
    try? h.write(to: URL(fileURLWithPath: outPath))
    FileHandle.standardError.write("skrev \(outPath) (\(dataLen) bytes, \(sr) Hz, callbacks=\(callbackCount), frames=\(totalFrames))\n".data(using: .utf8)!)
}

// SIGINT/SIGTERM -> skriv fil og avslutt
let sigSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigSrc.setEventHandler { cleanupAndWrite(); exit(0) }
sigSrc.resume()
signal(SIGINT, SIG_IGN)
let sigSrc2 = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
sigSrc2.setEventHandler { cleanupAndWrite(); exit(0) }
sigSrc2.resume()
signal(SIGTERM, SIG_IGN)

err = AudioDeviceStart(aggID, proc)
guard err == noErr else {
    FileHandle.standardError.write("AudioDeviceStart feilet: \(err)\n".data(using: .utf8)!)
    cleanupAndWrite(); exit(1)
}
FileHandle.standardError.write("tapper systemlyd (\(Int(sampleRate)) Hz, \(inChannels) ch) -> \(outPath)\n".data(using: .utf8)!)
dispatchMain()
