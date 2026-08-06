//
//  ContentView.swift
//  OmniVoice
//
//  Created by Rory Clear on 22/07/2026.
//

import SwiftUI
import Foundation
internal import Combine
import AVFoundation

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var buffers: [Int: MTLBuffer] = [:]
var buffer_sz: [Int: Int] = [:]
var programs: [String: MTLComputePipelineState] = [:]
var encode_graph: GraphRunner!
var model_graph: GraphRunner!
var model_graph2: GraphRunner!
var decode_graph: GraphRunner!
var generationProgress: Float = 0.0
var audioPlayer: AVAudioPlayer?
let CHAR_WEIGHTS = try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "char_weights", withExtension: "json")!))
let AUDIO_CHUNK_DURATION = 15.0
let FRAME_RATE = 25
let AUDIO_MASK_ID = 1024
let MAX_LEN = 500
let NUM_AUDIO_CODEBOOK = 8
let T_SHIFT = 0.1
let SAMPLING_RATE = 24_000
let CHUNK_SIZE = 960
let REF_AUDIO_LEN = 10
let tokenizer = Tokenizer()

let AUDIO_MASK_BUF = 1080
let ATTENTION_MASK_BUF = 1134
let TOKENS_BUF = 1136
let INPUT_IDS_BUF = 1135
let PRED_TOKENS_BUF = 1702

var delete_buffers: [String: [Int]] = [:]

class Tokenizer {
    let specialTokens: [String: Int32]
    let normalTokensBytes: [[UInt8]: Int32]

    private let byteDecoder: [Character: UInt8]

    init() {
        let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json")!
        let data = try! Data(contentsOf: url)
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Build locally first
        var decoder: [Character: UInt8] = [:]

        var bs = Array(33...126)
        bs += Array(161...172)
        bs += Array(174...255)

        var cs = bs
        var n = 0

        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        for (b, c) in zip(bs, cs) {
            decoder[Character(UnicodeScalar(c)!)] = UInt8(b)
        }

        // Special tokens
        let addedTokens = json["added_tokens"] as! [[String: Any]]

        let specials = Dictionary(
            uniqueKeysWithValues: addedTokens.map {
                (
                    $0["content"] as! String,
                    $0["id"] as! Int32
                )
            }
        )

        // Normal vocab
        let model = json["model"] as! [String: Any]
        let vocab = model["vocab"] as! [String: Int32]

        let normal = Dictionary(
            uniqueKeysWithValues: vocab.map { token, id in
                (
                    token.map { decoder[$0]! },
                    id
                )
            }
        )

        // Assign all properties last
        self.byteDecoder = decoder
        self.specialTokens = specials
        self.normalTokensBytes = normal
    }


    func encode(_ text: String) -> [Int32] {
        var result: [Int32] = []

        var remaining = text

        let specials = specialTokens.keys.sorted {
            $0.count > $1.count
        }

        while !remaining.isEmpty {

            if let special = specials.first(where: {
                remaining.hasPrefix($0)
            }) {
                result.append(specialTokens[special]!)
                remaining.removeFirst(special.count)
                continue
            }

            // encode until next possible special token
            var chunk = remaining

            if let next = specials.compactMap({
                remaining.range(of: $0)?.lowerBound
            }).min() {
                chunk = String(remaining[..<next])
                remaining = String(remaining[next...])
            } else {
                remaining = ""
            }

            result += encodeWord(chunk)
        }

        return result
    }


    private func encodeWord(_ word: String) -> [Int32] {
        let bytes = Array(word.utf8)

        if let id = normalTokensBytes[bytes] {
            return [id]
        }

        var parts = bytes.map {
            [$0]
        }

        while true {
            var bestID = Int32.max
            var bestIndex = -1

            for i in 0..<(parts.count - 1) {
                let merged = parts[i] + parts[i + 1]

                if let id = normalTokensBytes[merged], id < bestID {
                    bestID = id
                    bestIndex = i
                }
            }

            if bestIndex == -1 {
                break
            }

            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }

        return parts.map {
            normalTokensBytes[$0]!
        }
    }
}

class GraphRunner {
    let filename: String
    var calls: [[String: Any]] = []
    var copyouts: [Int] = []
    var copyins: [Int] = []
    var buffs: Set<Int> = []

    init(filename: String) {
        self.filename = filename
        print("GraphRunner initialized with:", filename)
        
        guard let url = Bundle.main.url(forResource: filename, withExtension: nil) else {
            print("File not found:", filename)
            return
        }

        autoreleasepool {
            do {
                var fileData: Data? = try Data(contentsOf: url)
                guard let data = fileData else { return }
                let json = try JSONSerialization.jsonObject(with: data, options: [])
                fileData = nil

                guard let items = json as? [Any] else {
                    print("Invalid JSON format")
                    return
                }

                for item in items {
                    autoreleasepool {
                        guard let dict = item as? [String: Any],
                              let key = dict.keys.first else {
                            return
                        }

                        if key == "buff_alloc" {
                            if let info = dict["buff_alloc"] as? [String: Any],
                               let num = info["num"] as? Int,
                               let size = info["size"] as? Int {

                                buffers[num] = device.makeBuffer(
                                    length: size,
                                    options: .storageModeShared
                                )

                                buffer_sz[num] = size
                            }

                        } else if key == "copyin" {
                            if let info = dict["copyin"] as? [String: Any],
                               let dest = info["dest"] as? Int,
                               let dataString = info["data"] as? String,
                               let decodedData = Data(base64Encoded: dataString),
                               let buffer = buffers[dest] {

                                copyins.append(dest)

                                let ptr = buffer.contents()
                                decodedData.copyBytes(
                                    to: ptr.assumingMemoryBound(to: UInt8.self),
                                    count: decodedData.count
                                )
                            }

                        } else if key == "program" {
                            if let info = dict["program"] as? [String: Any],
                               let name = info["name"] as? String,
                               let libString = info["lib"] as? String,
                               let libData = Data(base64Encoded: libString) {

                                let dispatchData = libData.withUnsafeBytes { ptr in
                                    DispatchData(bytes: ptr)
                                }

                                if let library = try? device.makeLibrary(
                                    data: dispatchData as! dispatch_data_t
                                ),
                                let function = library.makeFunction(name: name),
                                let pipeline = try? device.makeComputePipelineState(
                                    function: function
                                ) {
                                    programs[name] = pipeline
                                }
                            }

                        } else if key == "call" {
                            if let call = dict["call"] as? [String: Any] {
                                calls.append(call)
                                for buff in call["buffers"] as! [Int] {
                                    buffs.insert(buff)
                                }
                            }
                        } else if key == "copyout" {
                            if let copyout = dict["copyout"] as? Int {
                                copyouts.append(copyout)
                            }
                        }
                    }
                }
            } catch {
                print("Failed reading JSON:", error)
            }
        }
        
    }
    
    func run(vals_dict: [Int: Int]? = nil, globals_dict: [Int: Int]? = [:]) {
        autoreleasepool {
            let commandBuffer = queue.makeCommandBuffer()!
            for (index, item) in self.calls.enumerated() {
                let encoder = commandBuffer.makeComputeCommandEncoder()!
                print(index, "of", self.calls.count)
                let name = item["name"] as! String
                print(name)
                let pipeline = programs[name]!
                
                encoder.setComputePipelineState(pipeline)
                
                let bufferIDs = item["buffers"] as! [Int]
                let offsets = item["buffer_offsets"] as! [Int]
                let vals = item["vals"] as! [Int]
                print("vals =", vals)
                
                for i in 0..<bufferIDs.count {
                    let buffer = buffers[bufferIDs[i]]!
                    encoder.setBuffer(buffer, offset: offsets[i], index: i)
                }
                
                for i in 0..<vals.count{
                    var value = Int32(vals_dict![vals[i]]!)
                    encoder.setBytes(&value, length: 4, index: i+bufferIDs.count)
                }
                
                let global = item["global_size"] as! [Int]
                let local = item["local_size"] as! [Int]
                
                print("global_size =", global)
                
                let threadsPerGrid = MTLSize(
                    width: globals_dict?[global[0]] ?? global[0],
                    height: globals_dict?[global[1]] ?? global[1],
                    depth: globals_dict?[global[2]] ?? global[2]
                )
                
                
                let threadsPerThreadgroup = MTLSize(
                    width: local[0],
                    height: local[1],
                    depth: local[2]
                )
                
                encoder.dispatchThreadgroups(
                    threadsPerGrid,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
                encoder.endEncoding()
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
}

struct AddVoiceView: View {
    @Environment(\.dismiss) private var dismiss
    var onDismiss: () -> Void = {}
    @State private var audioURL: URL?
    @State private var transcript = ""
    @State private var voiceName = ""

    @State private var recorder: AVAudioRecorder?
    @State private var player: AVAudioPlayer?

    @State private var isRecording = false
    @State private var isPlaying = false
    @State private var errorMessage: String?
    @State private var recordingTimeLeft: Int = 10

    var canSubmit: Bool {
        audioURL != nil &&
        !voiceName.trimmingCharacters(in: .whitespaces).isEmpty &&
        !transcript.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var body: some View {
        VStack(spacing: 20) {

            HStack {
                Button {
                    startRecording()
                } label: {
                    Image(systemName: "mic.circle.fill")
                        .font(.largeTitle)
                        .foregroundColor(isRecording ? .red : .primary)
                }
                .disabled(isRecording)

                if isRecording {
                    Text("Recording \(recordingTimeLeft)s")
                        .foregroundColor(.red)
                }
            }

            if let audioURL {
                Button {
                    playRecording()
                } label: {
                    Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                }
            }

            if let errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
            }

            TextField("Voice name...", text: $voiceName)
                .textFieldStyle(.roundedBorder)
                .frame(width: 150)

            Text("Transcript:")

            TextEditor(text: $transcript)
                .frame(height: 80)
                .frame(maxWidth: .infinity)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.3))
                )
                #if os(iOS)
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder),
                                to: nil,
                                from: nil,
                                for: nil
                            )
                        }
                    }
                }
                #endif

            Button("Save Voice") {
                submitVoice(
                    audio: audioURL,
                    transcript: transcript,
                    name: voiceName
                )
            }
            .buttonStyle(.bordered)
            .disabled(!canSubmit)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("Add New Voice")
    }


    func startRecording() {
        requestMicrophonePermission()
        
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("temp.wav")
        audioURL = url
        
        // Record as raw PCM without any container
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 44100.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        
        do {
            recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder?.isMeteringEnabled = true
            recorder?.record()
            
            isRecording = true
            recordingTimeLeft = 10

            Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
                if self.recordingTimeLeft > 0 { self.recordingTimeLeft -= 1 }
                if self.recordingTimeLeft == 0 {
                    timer.invalidate()
                    self.stopRecording()
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stopRecording() { // ai slop that works
        recorder?.stop()
        recorder = nil
        isRecording = false
        if let url = audioURL {
            do {
                var audioData = try Data(contentsOf: url)
                if let dataRange = audioData.range(of: "data".data(using: .ascii)!) {
                    let dataStart = dataRange.upperBound + 4
                    audioData = audioData.subdata(in: dataStart..<audioData.count)
                }
                let sampleRate: UInt32 = 44100
                let channels: UInt16 = 1
                let bitsPerSample: UInt16 = 16
                let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
                let blockAlign = channels * (bitsPerSample / 8)
                let dataSize = UInt32(audioData.count)
                let fileSize = 36 + dataSize
                
                var wavData = Data()
                
                // RIFF header
                wavData.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
                var fs = fileSize.littleEndian
                wavData.append(Data(bytes: &fs, count: 4))
                wavData.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"
                
                // fmt chunk
                wavData.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
                var chunkSize: UInt32 = 16
                wavData.append(Data(bytes: &chunkSize, count: 4))
                var format: UInt16 = 1 // PCM
                wavData.append(Data(bytes: &format, count: 2))
                var ch = channels
                wavData.append(Data(bytes: &ch, count: 2))
                var sr = sampleRate
                wavData.append(Data(bytes: &sr, count: 4))
                var br = byteRate
                wavData.append(Data(bytes: &br, count: 4))
                var ba = blockAlign
                wavData.append(Data(bytes: &ba, count: 2))
                var bps = bitsPerSample
                wavData.append(Data(bytes: &bps, count: 2))
                
                // data chunk
                wavData.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
                var ds = dataSize
                wavData.append(Data(bytes: &ds, count: 4))
                wavData.append(audioData)
                
                try wavData.write(to: url)
                DispatchQueue.main.async {
                    self.playRecording()
                }
                
            } catch {
                print("WAV writing error: \(error)")
            }
        }
    }


    func playRecording() {
        guard let audioURL else { return }

        do {
            if isPlaying {
                player?.stop()
                isPlaying = false
                return
            }

            player = try AVAudioPlayer(contentsOf: audioURL)
            player?.play()
            isPlaying = true

            DispatchQueue.main.asyncAfter(
                deadline: .now() + (player?.duration ?? 0)
            ) {
                isPlaying = false
            }

        } catch {
            errorMessage = error.localizedDescription
        }
    }


    func submitVoice(audio: URL?, transcript: String, name: String) {
        guard let audio else {
            print("missing audio")  
            return
        }
        do {
            let audioData = try Data(contentsOf: audio)
            let base64Audio = audioData.base64EncodedString()

            let json: [String: String] = [
                "ref_text": transcript,
                "ref_audio": base64Audio
            ]

            let data = try JSONSerialization.data(
                withJSONObject: json,
                options: [.prettyPrinted]
            )

            let documentsURL = FileManager.default.urls(
                for: .documentDirectory,
                in: .userDomainMask
            )[0]

            let fileURL = documentsURL.appendingPathComponent("\(name).cv")
            try data.write(to: fileURL)
            onDismiss()
            dismiss()

        } catch {
            print("Failed saving CV:", error.localizedDescription)
        }
    }


    func requestMicrophonePermission() {
    #if os(iOS)
        AVAudioApplication.requestRecordPermission { granted in
            if !granted {
                DispatchQueue.main.async {
                    errorMessage = "Microphone permission denied."
                }
            }
        }
        do {
            let session = AVAudioSession.sharedInstance()

            try session.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.defaultToSpeaker]
            )

            try session.setActive(true)

        } catch {
            errorMessage = error.localizedDescription
        }
    #endif
    }
}

struct VoiceListView: View {
    @Binding var voices: [URL]
    @Binding var selectedVoice: URL?
    
    private var filteredVoices: [URL] {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return voices.filter { $0.path.contains(documents.path) }
    }

    var body: some View {
        VStack {
            Group {
                #if os(macOS)
                Text("Right-click to delete a voice")
                #else
                Text("Swipe left to delete a voice")
                #endif
            }
            .font(.footnote)
            .foregroundColor(.secondary)

            List {
                ForEach(filteredVoices, id: \.self) { url in
                    row(for: url)
                }
                .onDelete(perform: deleteOffsets)
            }
        }
        .navigationTitle("Manage Voices")
    }
    
    
    @ViewBuilder
    private func row(for url: URL) -> some View {
        HStack {
            Text(url.deletingPathExtension().lastPathComponent)
            Spacer()
            if selectedVoice == url {
                Image(systemName: "checkmark")
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            selectedVoice = url
        }
        .contextMenu {
            Button("Delete", role: .destructive) {
                delete(url)
            }
        }
    }

    func delete(_ url: URL) {
        deleteURLs([url])
    }
    func deleteOffsets(at offsets: IndexSet) {
        let urls = offsets.map { filteredVoices[$0] }
        deleteURLs(urls)
    }

    func deleteURLs(_ urls: [URL]) {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        for url in urls {
            if url.path.contains(documents.path) {
                try? FileManager.default.removeItem(at: url)
            }
        }

        voices.removeAll { urls.contains($0) }

        if let selected = selectedVoice, !voices.contains(selected) {
            selectedVoice = voices.first
        }
    }
}

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var voices: [URL] = []
    @State private var selectedVoice: URL?
    @State private var languages: [Language] = []
    @State private var selectedLanguage: String = "None"
    @State private var progress: Float = 0.0
    @State private var isGenerating: Bool = false
    @State private var showPlayer: Bool = false
    @State private var showRAMWarning: Bool = false
    @State private var didCheckRAM: Bool = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image("icon")
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 25))
                    .frame(width: 120, height: 120)
                
                VStack(alignment: .leading, spacing: 6) {
                    Text("Enter Text:")
                        .font(.headline)

                    TextEditor(text: $inputText)
                        .frame(width: 300, height: 100)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.4))
                        )
                        #if os(iOS)
                        .toolbar {
                            ToolbarItemGroup(placement: .keyboard) {
                                Spacer()
                                Button("Done") {
                                    UIApplication.shared.sendAction(
                                        #selector(UIResponder.resignFirstResponder),
                                        to: nil,
                                        from: nil,
                                        for: nil
                                    )
                                }
                            }
                        }
                        #endif
                }
                .padding(.horizontal)
                
                HStack {
                    Picker("Voice", selection: $selectedVoice) {
                        ForEach(voices, id: \.self) { url in
                            Text(url.deletingPathExtension().lastPathComponent)
                                .tag(Optional(url))
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.4))
                    )

                    NavigationLink("Manage Voices") {
                        VoiceListView(
                            voices: $voices,
                            selectedVoice: $selectedVoice
                        )
                    }
                    .buttonStyle(.bordered)
                }
                
                NavigationLink(destination: AddVoiceView(onDismiss: {
                    loadVoices()
                })) {
                    Text("Add New Voice")
                        .frame(width: 160)
                }
                .buttonStyle(.bordered)
                
                Picker("", selection: $selectedLanguage) {
                    Text("Auto Language").tag("None")
                    ForEach(languages) { language in
                        Text(language.name).tag(language.id)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 250)
                .padding(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.4))
                )
                .padding(.horizontal)
                .frame(width: 250)
                .padding(.horizontal)

                if isGenerating {
                    ProgressView(value: progress)
                        .padding(.horizontal)
                    Text("\(Int(progress * 100))%")
                }

                if showPlayer {
                    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                        .appendingPathComponent("output.wav")

                    HStack {
                        Button(action: { playAudio(url) }) {
                            Image(systemName: "play.circle.fill")
                                .font(.largeTitle)
                        }

                        ShareLink(item: url) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.title2)
                        }
                    }
                }

                Button("Generate Audio") {
                    showPlayer = false
                    isGenerating = true
                    progress = 0

                    Task.detached {
                        generate(
                            text: inputText,
                            cvFile: selectedVoice!.path,
                            num_steps: 32,
                            language: selectedLanguage
                        )
                        generationProgress = 1.0
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isGenerating)
            }
            .padding()
            .onReceive(Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()) { _ in
                if isGenerating {
                    progress = generationProgress
                    if generationProgress >= 1.0 {
                        isGenerating = false
                        generationProgress = 0
                        showPlayer = true
                        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                            .appendingPathComponent("output.wav")
                        playAudio(url)
                    }
                }
            }
            .onAppear {
                loadVoices()
                loadLanguages()
                if !didCheckRAM {
                    checkRAM()
                    didCheckRAM = true
                }
            }
            .alert("Memory Warning", isPresented: $showRAMWarning) {
                Button("OK") {}
            } message: {
                Text("6GB of available RAM required to run the voice model to avoid crashes.")
            }
            #if os(iOS)
            .padding()
            .navigationTitle("ClearVoice")
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
    }
    
    func checkRAM() {
        #if os(iOS)
        let availableGB = Double(os_proc_available_memory()) / (1024.0 * 1024.0 * 1024.0)
        if availableGB < 6.0 {
            showRAMWarning = true
        }
        #endif
    }
    
    func playAudio(_ url: URL) {
        audioPlayer = try? AVAudioPlayer(contentsOf: url)
        audioPlayer?.play()
    }
    

    func loadVoices() {
        var found: [URL] = []
        let fm = FileManager.default
        if let bundleURLs = Bundle.main.urls(forResourcesWithExtension: "cv", subdirectory: nil) { found.append(contentsOf: bundleURLs) }

        if let documents = try? fm.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ),
        let files = try? fm.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil) {
            found.append(contentsOf: files.filter { $0.pathExtension == "cv" })
        }
        voices = found
        if selectedVoice == nil { selectedVoice = voices.first }
    }
    
    struct Language: Identifiable, Codable {
        let id: String
        let name: String
    }
    
    func loadLanguages() {
        guard let url = Bundle.main.url(
            forResource: "languages",
            withExtension: "json"
        ) else {
            return
        }

        do {
            let data = try Data(contentsOf: url)
            languages = try JSONDecoder().decode(
                [Language].self,
                from: data
            )
        } catch {
            print("Failed to load languages:", error)
        }
    }
}

struct CVFile: Decodable {
    let ref_text: String
    let ref_audio: String // base64 encoded wav
}

func generate(text: String, cvFile: String, num_steps: Int, language: String) {
    if let url = Bundle.main.url(forResource: "buffers", withExtension: "json"),
       let data = try? Data(contentsOf: url),
       let dict = try? JSONDecoder().decode([String: [Int]].self, from: data) {
        delete_buffers = dict
    }
    
    encode_graph = GraphRunner(filename: "0.rc")
    let url = URL(fileURLWithPath: cvFile)
    guard let data = try? Data(contentsOf: url) else { fatalError("Failed to load CV file at path: \(cvFile)")}

    let decoder = JSONDecoder()
    guard let cv = try? decoder.decode(CVFile.self, from: data) else { fatalError("Failed to decode CV JSON")}

    let refText = cv.ref_text
    var ref_wav = loadAudioFromBase64(cv.ref_audio, samplingRate: 24000)
    let wav_len = ref_wav.count
    ref_wav = expandWav(ref_wav)
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), ref_wav, ref_wav.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    var ref_audio_tokens = get_ref_tokens()
    // free up some memory
    for b in delete_buffers["0.rc"]! { buffers[b] = nil }
    model_graph = GraphRunner(filename: "1.rc")
    model_graph2 = GraphRunner(filename: "2.rc")
    //todo shrink graphs (spread the allocs to where needed?
    ref_audio_tokens = ref_audio_tokens .map { Array($0.prefix(wav_len / CHUNK_SIZE)) }
    let styleTokens = tokenizer.encode("<|denoise|><|lang_start|>\(language)<|lang_end|><|instruct_start|>None<|instruct_end|>")
    let chunks = getChunks(text: text, refText: refText, wavLen: wav_len, styleTokens: styleTokens, num_ref_tokens: Int(wav_len / CHUNK_SIZE))
    var rets: [[[Int32]]] = []
    var target_lengths: [Int] = []
    for (idx, chunk) in chunks.enumerated() {
        var tokens: [[Int32]] = Array(repeating: Array(repeating: Int32(AUDIO_MASK_ID), count: MAX_LEN), count: NUM_AUDIO_CODEBOOK)
        let target_length = estimateTargetTokens(text: chunk, refText: refText, numRefAudioTokens: ref_audio_tokens[0].count)
        target_lengths.append(target_length)
        let (sched, num_steps) = get_sched(numSteps: num_steps, targetLength: target_length)
        let combined = [refText, chunk].map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: " ")
        let text_tokens = tokenizer.encode("<|text_start|>\(combined)<|text_end|>").map { Int32($0) }
        var (c_len, audio_mask, attention_mask, input_ids) = getInputs(textTokens: text_tokens, targetLength: target_length, refAudioTokens: ref_audio_tokens, styleTokens: styleTokens)
        for step in 0..<num_steps {
            generationProgress = Float(step + idx * num_steps) / Float(chunks.count * num_steps)
            //copyins
            
            let input_ids_flat = input_ids.flatMap { $0.flatMap { $0 } }
            buffers[INPUT_IDS_BUF]!.contents().copyMemory(from: input_ids_flat, byteCount: input_ids_flat.count * MemoryLayout<Int32>.stride)
            
            let attention_mask_flat = attention_mask.flatMap { $0.flatMap { $0.flatMap { $0 } } }
            buffers[ATTENTION_MASK_BUF]!.contents().copyMemory(from: attention_mask_flat, byteCount: attention_mask_flat.count)
            
            let tokens_flat = tokens.flatMap { $0 }
            buffers[TOKENS_BUF]!.contents().copyMemory(from: tokens_flat, byteCount: tokens_flat.count * MemoryLayout<Int32>.stride)
            
            let audio_mask_flat = audio_mask.flatMap { $0 }
            buffers[AUDIO_MASK_BUF]!.contents().copyMemory(from: audio_mask_flat, byteCount: audio_mask_flat.count)
            
            if (step == 0) {
                model_graph.run(vals_dict: [113: target_length ,373: c_len], globals_dict: [113: target_length ,373: c_len, 373*2: c_len*2])
            } else {
                model_graph2.run(vals_dict: [113: target_length ,373: c_len], globals_dict: [113: target_length ,373: c_len, 373*2: c_len*2])
            }
            let scores_out = Array(UnsafeBufferPointer(start: buffers[model_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Float32.self), count: buffer_sz[model_graph.copyouts[0]]! / 4))[0..<(NUM_AUDIO_CODEBOOK * MAX_LEN)]
            let n = scores_out.count / NUM_AUDIO_CODEBOOK
            var scores = stride(from: 0, to: scores_out.count, by: n).map { Array(scores_out[$0..<min($0 + n, scores_out.count)])}
            
            let pred_tokens_out = Array(UnsafeBufferPointer(start: buffers[PRED_TOKENS_BUF]!.contents().assumingMemoryBound(to: Float32.self), count: buffer_sz[PRED_TOKENS_BUF]! / 4))[0..<(MAX_LEN * NUM_AUDIO_CODEBOOK)]
            var pred_tokens = (0..<NUM_AUDIO_CODEBOOK).map { i in Array(pred_tokens_out[(i * MAX_LEN)..<((i + 1) * MAX_LEN)])}
            
            
            scores = scores.map { Array($0.prefix(target_length)) }
            pred_tokens = pred_tokens.map { Array($0.prefix(target_length)) }
                
            let flatScores = scores.flatMap { $0 }
            let sortedIdx = flatScores.indices.sorted { flatScores[$0] > flatScores[$1]}
            let topkIdx = Array(sortedIdx.prefix(sched[step]))
            
            var sampleTokens = tokens.map { Array($0.prefix(target_length)) }
            sampleTokens = sampleTokens.compactMap { $0 }
            
            let predFlat = pred_tokens.flatMap { $0 }
            var sampleTokensFlat = sampleTokens.flatMap { $0 }
            for (_, idx) in topkIdx.enumerated() { sampleTokensFlat[idx] = Int32(predFlat[idx]) }
            
            sampleTokens = stride(from: 0, to: sampleTokensFlat.count, by: target_length).map { Array(sampleTokensFlat[$0..<($0 + target_length)]) }
            
            for i in 0..<NUM_AUDIO_CODEBOOK { for j in 0..<target_length { tokens[i][j] = sampleTokens[i][j] } }
            print(sampleTokens)
            
            for i in 0..<NUM_AUDIO_CODEBOOK {
                for j in 0..<target_length {
                    input_ids[0][i][c_len - target_length + j] = sampleTokens[i][j]
                    input_ids[1][i][j] = sampleTokens[i][j]
                }
            }
        }
        rets.append(tokens)
    }
    
    // todo move and copyin
    // free up some memory
    for b in delete_buffers["1.rc"]! { buffers[b] = nil }
    for b in delete_buffers["2.rc"]! { buffers[b] = nil }
    decode_graph = GraphRunner(filename: "100.rc")
    var combinedWaveform: [Float] = []
    for (i, ret) in rets.enumerated() {
        let flatRet = ret.flatMap { $0 }
        buffers[TOKENS_BUF]!.contents().copyMemory(from: flatRet, byteCount: flatRet.count * 4)
        decode_graph.run()
        let wv = Array(Array(UnsafeBufferPointer(start: buffers[decode_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Float32.self),count: buffer_sz[decode_graph.copyouts[0]]! / 4))[0..<(target_lengths[i] * CHUNK_SIZE)])
        combinedWaveform.append(contentsOf: wv)
    }
    
    let wavData = waveformToWavBytes(audio: combinedWaveform, sampleRate: SAMPLING_RATE)
    
    let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("output.wav")

    do {
        try wavData.write(to: fileURL)
        print("Saved to \(fileURL.path)")
    } catch {
        print("Error writing file: \(error)")
    }
    
    print("rory rets =",rets)
    print("1")
    
    buffers.removeAll()
    buffer_sz.removeAll()
    programs.removeAll()
    encode_graph = nil
    model_graph = nil
    model_graph2 = nil
    decode_graph = nil
}

func waveformToWavBytes(audio: [Float], sampleRate: Int) -> Data {
    let channels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let bytesPerSample = Int(bitsPerSample / 8)

    // Clip and convert to Int16
    let audioInt16: [Int16] = audio.map {
        let clipped = max(-1.0, min(1.0, $0))
        return Int16(clipped * 32767.0)
    }

    let byteRate = UInt32(sampleRate) * UInt32(channels) * UInt32(bytesPerSample)
    let blockAlign = channels * UInt16(bytesPerSample)
    let dataSize = UInt32(audioInt16.count * bytesPerSample)
    let chunkSize = UInt32(36) + dataSize

    var data = Data()

    // Helper to append values as little-endian
    func append<T>(_ value: T) {
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    // RIFF header
    data.append("RIFF".data(using: .ascii)!)
    append(chunkSize)
    data.append("WAVE".data(using: .ascii)!)

    // fmt chunk
    data.append("fmt ".data(using: .ascii)!)
    append(UInt32(16))                  // Subchunk1Size
    append(UInt16(1))                   // PCM format
    append(channels)
    append(UInt32(sampleRate))
    append(byteRate)
    append(blockAlign)
    append(bitsPerSample)

    // data chunk
    data.append("data".data(using: .ascii)!)
    append(dataSize)

    // Audio samples
    audioInt16.forEach { append($0) }

    return data
}

func get_sched(numSteps: Int, targetLength: Int) -> ([Int], Int) {
    let timesteps = (0...numSteps).map { i -> Double in
        Double(i) / Double(numSteps)
    }.map { t -> Double in
        (T_SHIFT * t) / (1 + (T_SHIFT - 1) * t)
    }

    let totalMask = targetLength * NUM_AUDIO_CODEBOOK
    var rem = totalMask
    var sched: [Int] = []

    for step in 0..<numSteps {
        let num: Int

        if step == numSteps - 1 {
            num = rem
        } else {
            num = min(
                Int(ceil(Double(totalMask) * (timesteps[step + 1] - timesteps[step]))),
                rem
            )
        }
        sched.append(num)
        if num >= MAX_LEN {
            return get_sched(
                numSteps: numSteps * 2,
                targetLength: targetLength
            )
        }
        rem -= num
    }
    return (sched, numSteps)
}

func loadAudioFromBase64(_ base64: String, samplingRate: Int = SAMPLING_RATE) -> [Float] {
    guard let audioData = Data(base64Encoded: base64) else {
        fatalError("Invalid base64 audio")
    }

    let (data, sr) = load_waveform(audioData)

    let sampleCount = data[0].count
    var mono = [Float]()
    mono.reserveCapacity(sampleCount)

    for i in 0..<sampleCount {
        var sum: Float = 0
        for channel in data {
            sum += channel[i]
        }
        mono.append(sum / Float(data.count))
    }
    let resampled = resample([mono], origSR: sr, targetSR: samplingRate)[0]
    let rms = sqrt(
        resampled.reduce(0.0) { $0 + Double($1 * $1) } /
        Double(resampled.count)
    )
    var output = resampled
    if rms > 0 && rms < 0.1 {
        let scale = Float(0.1 / rms)
        output = output.map { $0 * scale }
    }
    return output
}

func getInputs(textTokens: [Int32], targetLength: Int, refAudioTokens: [[Int32]], styleTokens: [Int32]) -> (Int, [[Bool]], [[[[Bool]]]], [[[Int32]]]) {
    let targetAudioTokens = Array(repeating: Int32(AUDIO_MASK_ID), count: targetLength)
    var c_len = styleTokens.count + textTokens.count + refAudioTokens[0].count + targetLength
    let condAudioStartIdx = c_len - targetLength - refAudioTokens[0].count

    var condinput_ids: [[[Int32]]] = [[]]
    for i in 0..<NUM_AUDIO_CODEBOOK { condinput_ids[0].append(styleTokens + textTokens + refAudioTokens[i] + targetAudioTokens) }
    var input_ids = Array(
        repeating: Array(
            repeating: Array(repeating: Int32(AUDIO_MASK_ID), count: MAX_LEN),
            count: NUM_AUDIO_CODEBOOK
        ),
        count: 2
    )

    for i in 0..<NUM_AUDIO_CODEBOOK {for j in 0..<c_len {input_ids[0][i][j] = condinput_ids[0][i][j]}}
    for i in 0..<NUM_AUDIO_CODEBOOK {
        let start = condinput_ids[0][i].count - targetLength
        for j in 0..<targetLength { input_ids[1][i][j] = condinput_ids[0][i][start + j] }
    }

    let condaudio_mask = Array(repeating: false, count: condAudioStartIdx) + Array(repeating: true, count: c_len - condAudioStartIdx)

    var audio_mask = Array(repeating: Array(repeating: false, count: MAX_LEN), count: 2)
    for i in 0..<c_len {audio_mask[0][i] = condaudio_mask[i]}
    for i in 0..<targetLength { audio_mask[1][i] = condaudio_mask[condaudio_mask.count - targetLength + i]}

    var attentionMask = Array(
        repeating: Array(
            repeating: Array(
                repeating: Array(repeating: false, count: MAX_LEN),
                count: MAX_LEN
            ),
            count: 1
        ),
        count: 2
    )
    for i in 0..<c_len { for j in 0..<c_len { attentionMask[0][0][i][j] = true } }
    for i in 0..<targetLength { for j in 0..<targetLength { attentionMask[1][0][i][j] = true }}
    for i in targetLength..<c_len { attentionMask[1][0][i][i] = true}

    return (c_len, audio_mask, attentionMask, input_ids)
}

func expandWav(_ refWav: [Float], ref_audio_length: Int = REF_AUDIO_LEN) -> [Float] {
    var refWav = refWav
    let clipSize = refWav.count % CHUNK_SIZE

    if clipSize > 0 {
        refWav = Array(refWav.dropLast(clipSize))
    }

    let wavLen = refWav.count
    let targetLen = SAMPLING_RATE * ref_audio_length

    if wavLen <= targetLen {
        refWav += Array(repeating: 0.0, count: targetLen - wavLen)
    } else {
        refWav = Array(refWav.prefix(targetLen))
    }

    return refWav
}

func resample(
    _ data: [[Float]],
    origSR: Int,
    targetSR: Int
) -> [[Float]] {
    let sampleCount = data[0].count

    let duration = Double(sampleCount) / Double(origSR)

    let origTimes = (0..<sampleCount).map {
        Double($0) * duration / Double(sampleCount)
    }

    let newLength = Int(duration * Double(targetSR))

    let newTimes = (0..<newLength).map {
        Double($0) * duration / Double(newLength)
    }

    return data.map { channel in
        interp1D(
            newTimes: newTimes,
            origTimes: origTimes,
            values: channel
        )
    }
}

func interp1D(
    newTimes: [Double],
    origTimes: [Double],
    values: [Float]
) -> [Float] {
    var result = [Float]()
    result.reserveCapacity(newTimes.count)

    var index = 0

    for t in newTimes {
        while index < origTimes.count - 2 &&
              origTimes[index + 1] < t {
            index += 1
        }

        let t0 = origTimes[index]
        let t1 = origTimes[index + 1]

        let y0 = Double(values[index])
        let y1 = Double(values[index + 1])

        let ratio = (t - t0) / (t1 - t0)

        result.append(
            Float(y0 + ratio * (y1 - y0))
        )
    }

    return result
}

func load_waveform(_ data: Data) -> ([[Float]], Int) {
    let bytes = [UInt8](data)
    let sampleRate = Int(readUInt32LE(bytes, offset: 24))
    let channels = Int(readUInt16LE(bytes, offset: 22))
    let dataMarker = [UInt8]([100, 97, 116, 97]) // "data"
    var dataOffset = 0

    for i in 0..<(bytes.count - 4) {
        if Array(bytes[i..<i+4]) == dataMarker {
            dataOffset = i + 8
            break
        }
    }

    let rawSamples = Array(bytes[dataOffset..<bytes.count])

    // int16 = 2 bytes
    let nSamples = rawSamples.count / 2

    var samples = [Int16]()
    samples.reserveCapacity(nSamples)

    for i in stride(from: 0, to: rawSamples.count, by: 2) {
        let sample = Int16(bitPattern:
            UInt16(rawSamples[i]) |
            (UInt16(rawSamples[i + 1]) << 8)
        )
        samples.append(sample)
    }

    // Create channel arrays
    var audio = Array(
        repeating: [Float](),
        count: channels
    )

    for i in stride(from: 0, to: samples.count, by: channels) {
        for ch in 0..<channels {
            if i + ch < samples.count {
                audio[ch].append(
                    Float(samples[i + ch]) / 32768.0
                )
            }
        }
    }

    return (audio, sampleRate)
}

func readUInt16LE(_ bytes: [UInt8], offset: Int) -> UInt16 {
    return UInt16(bytes[offset]) |
           (UInt16(bytes[offset + 1]) << 8)
}

func readUInt32LE(_ bytes: [UInt8], offset: Int) -> UInt32 {
    return UInt32(bytes[offset]) |
           (UInt32(bytes[offset + 1]) << 8) |
           (UInt32(bytes[offset + 2]) << 16) |
           (UInt32(bytes[offset + 3]) << 24)
}

#Preview {
    ContentView()
}

func get_ref_tokens(ref_audio_length: Int = REF_AUDIO_LEN) -> [[Int32]] {
    let count = buffer_sz[encode_graph.copyouts[0]]! / 4
    let flat = Array(UnsafeBufferPointer(start: buffers[encode_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Int32.self), count: count))
    let prefixCount = Int((REF_AUDIO_LEN * SAMPLING_RATE * NUM_AUDIO_CODEBOOK) / CHUNK_SIZE)
    let trimmed = Array(flat.prefix(prefixCount))
    let n = trimmed.count / NUM_AUDIO_CODEBOOK
    return stride(from: 0, to: trimmed.count, by: n).map { Array(trimmed[$0..<($0 + n)]) }
}

func getChunks(text: String, refText: String, wavLen: Int, styleTokens: [Int32], num_ref_tokens: Int) -> [String] {
    let pattern = #"[^。，！？；：、,.?，\(\)\[\]（）【】]+[。，！？；：、,.?，\(\)\[\]（）【】]?"#
    let regex = try! NSRegularExpression(pattern: pattern, options: [])

    let nsText = text as NSString
    let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length))

    let chunksSmall: [String] = matches.map { nsText.substring(with: $0.range) }

    var chunks: [String] = [""]
    var j = 0

    for i in 0..<chunksSmall.count {
        let combined = chunks[j] + chunksSmall[i]
        
        let targetLength = estimateTargetTokens(text: combined, refText: refText, numRefAudioTokens: Int(wavLen / CHUNK_SIZE))
        print(combined, targetLength)

        let joinedText = [refText, combined].map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")

        let textTokens = tokenizer.encode("<|text_start|>\(joinedText)<|text_end|>")
        print(styleTokens.count + textTokens.count + num_ref_tokens + targetLength)
        if styleTokens.count + textTokens.count + num_ref_tokens + targetLength < MAX_LEN {
            chunks[j] += chunksSmall[i]
        } else {
            // Check if the single chunk is too large
            let singleTargetLength = estimateTargetTokens(text: chunksSmall[i], refText: refText, numRefAudioTokens: Int(wavLen / CHUNK_SIZE))
            let singleJoinedText = [refText, chunksSmall[i]].map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")
            let singleTextTokens = tokenizer.encode("<|text_start|>\(singleJoinedText)<|text_end|>")
            if styleTokens.count + singleTextTokens.count + num_ref_tokens + singleTargetLength > MAX_LEN {
                let words = chunksSmall[i].components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                var currentChunk = ""
                for word in words {
                    let testChunk = currentChunk.isEmpty ? word : currentChunk + " " + word
                    let testTargetLength = estimateTargetTokens(text: testChunk, refText: refText, numRefAudioTokens: Int(wavLen / CHUNK_SIZE))
                    let testJoinedText = [refText, testChunk].map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")
                    let testTextTokens = tokenizer.encode("<|text_start|>\(testJoinedText)<|text_end|>")
                    
                    if styleTokens.count + testTextTokens.count + num_ref_tokens + testTargetLength < MAX_LEN {
                        currentChunk = testChunk
                    } else {
                        if !currentChunk.isEmpty {
                            chunks.append(currentChunk)
                            j += 1
                        }
                        currentChunk = word
                    }
                }
                
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                    j += 1
                }
            } else {
                chunks.append(chunksSmall[i])
                j += 1
            }
        }
    }
    print(chunks)
    return chunks
}

func estimateTargetTokens(text: String, refText: String, numRefAudioTokens: Int,) -> Int {
    func weightSum(for string: String) -> Double {return string.unicodeScalars.reduce(0.0) { sum, scalar in sum + Double(CHAR_WEIGHTS[Int(scalar.value)]) }}
    let refWeight = weightSum(for: refText)
    let speedFactor = refWeight / Double(numRefAudioTokens)
    let targetWeight = weightSum(for: text)
    let estimatedDuration = targetWeight / speedFactor
    return Int(estimatedDuration)
}



