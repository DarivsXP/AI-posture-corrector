<script setup>
import AuthenticatedLayout from '@/Layouts/AuthenticatedLayout.vue';
import { ref, onUnmounted } from 'vue';
import axios from 'axios';

// --- State ---
const videoRef = ref(null);
const isRunning = ref(false);
const currentScore = ref(0); // Start at 0 to show we haven't calculated yet
const isSlouching = ref(false);
const statusMessage = ref("Ready");
const connectionStatus = ref("Waiting..."); // New debug status
const angles = ref({ torso: 0, neck: 0, back: 0 });

// --- Timers ---
let processInterval = null;
let feedbackTimer = null;
let sessionData = {
    scores: [],
    slouchFrames: 0,
    totalFrames: 0,
    alerts: 0
};

// --- 1. Camera Logic ---
const startCamera = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.value) {
            videoRef.value.srcObject = stream;
        }
        isRunning.value = true;
        connectionStatus.value = "Camera Active. Connecting to AI...";

        // Start Processing Loop (Every 500ms)
        processInterval = setInterval(processFrame, 500);

        // Start Data Upload Loop (Every 30 seconds)
        setInterval(uploadSessionData, 30000);

    } catch (err) {
        console.error("Camera Error:", err);
        alert("Could not access camera. Please allow permissions.");
    }
};

const stopCamera = () => {
    isRunning.value = false;
    clearInterval(processInterval);
    if (videoRef.value && videoRef.value.srcObject) {
        videoRef.value.srcObject.getTracks().forEach(track => track.stop());
    }
    connectionStatus.value = "Stopped";
};

// --- 2. AI Processing Logic ---
const processFrame = async () => {
    if (!videoRef.value) return;

    // Capture frame to canvas
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.value.videoWidth;
    canvas.height = videoRef.value.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.value, 0, 0);

    // Convert to Blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob);

        try {
            // Send to Python Server (Ensure server.py is running on port 5000)
            const res = await axios.post('http://127.0.0.1:5000/process_frame', formData);

            // Update UI with Real Data
            connectionStatus.value = "AI Connected (Processing)";
            currentScore.value = res.data.score;
            isSlouching.value = res.data.is_slouching;
            angles.value = res.data.angles;

            // Update Session Stats
            sessionData.scores.push(res.data.score);
            sessionData.totalFrames++;
            if (res.data.is_slouching) sessionData.slouchFrames++;

            // Run Feedback Logic
            handleAdaptiveFeedback(res.data.is_slouching);

        } catch (err) {
            console.error("AI Server Error:", err);
            connectionStatus.value = "Error: Is Python Server Running?";
        }
    }, 'image/jpeg');
};

// --- 3. Feedback Logic ---
let slouchStartTime = null;

const handleAdaptiveFeedback = (slouching) => {
    if (slouching) {
        if (!slouchStartTime) slouchStartTime = Date.now();

        const duration = (Date.now() - slouchStartTime) / 1000;

        if (duration > 15) {
            statusMessage.value = "CRITICAL: Sit Up!";
            sessionData.alerts++;
            slouchStartTime = Date.now();
        } else if (duration > 5) {
            statusMessage.value = "Warning...";
        } else {
            statusMessage.value = "Poor Posture";
        }
    } else {
        slouchStartTime = null;
        statusMessage.value = "Good Posture";
    }
};

// --- 4. Data Logging ---
const uploadSessionData = () => {
    if (sessionData.totalFrames === 0) return;

    const avgScore = Math.round(sessionData.scores.reduce((a, b) => a + b, 0) / sessionData.scores.length);
    const slouchTime = Math.round((sessionData.slouchFrames / sessionData.totalFrames) * 30);

    const payload = {
        score: avgScore,
        slouch_duration: slouchTime,
        alert_count: sessionData.alerts
    };

    axios.post('/posture-chunks', payload)
        .then(() => console.log("Data Saved to Dashboard"))
        .catch(err => console.error("Save Failed", err));

    sessionData = { scores: [], slouchFrames: 0, totalFrames: 0, alerts: 0 };
};

onUnmounted(() => stopCamera());
</script>

<template>
    <AuthenticatedLayout>
        <template #header>
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">Live Monitoring</h2>
        </template>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6 flex flex-col items-center">

                    <!-- Status Bar -->
                    <div class="w-full mb-4 p-2 rounded text-center text-sm font-mono"
                         :class="connectionStatus.includes('Error') ? 'bg-red-100 text-red-700' : 'bg-blue-50 text-blue-700'">
                        System Status: {{ connectionStatus }}
                    </div>

                    <!-- Main Camera View -->
                    <div class="relative bg-black rounded-lg shadow-lg overflow-hidden" style="width: 640px; height: 480px;">
                        <video ref="videoRef" autoplay playsinline class="w-full h-full object-cover"></video>

                        <!-- Overlay HUD -->
                        <div class="absolute top-4 left-4 bg-black/60 text-white p-4 rounded backdrop-blur-sm border border-white/10">
                            <div class="text-2xl font-bold mb-1" :class="currentScore >= 70 ? 'text-green-400' : 'text-red-400'">
                                Score: {{ currentScore }}%
                            </div>
                            <div class="font-semibold text-lg mb-2" :class="isSlouching ? 'text-red-300' : 'text-green-300'">
                                {{ statusMessage }}
                            </div>

                            <!-- Debug Angles -->
                            <div class="text-xs text-gray-400 font-mono space-y-1 border-t border-gray-500/50 pt-2 mt-2">
                                <div>Torso: {{ angles.torso }}°</div>
                                <div>Neck:  {{ angles.neck }}°</div>
                                <div>Back:  {{ angles.back }}°</div>
                            </div>
                        </div>
                    </div>

                    <!-- Controls -->
                    <div class="mt-8 flex gap-4">
                        <button @click="startCamera" v-if="!isRunning"
                                class="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow hover:bg-blue-700 transition flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" /></svg>
                            Start Monitoring
                        </button>

                        <button @click="stopCamera" v-else
                                class="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow hover:bg-red-700 transition flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd" /></svg>
                            Stop Session
                        </button>
                    </div>

                </div>
            </div>
        </div>
    </AuthenticatedLayout>
</template>
