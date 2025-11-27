<script setup>
import AuthenticatedLayout from '@/Layouts/AuthenticatedLayout.vue';
import { Head, usePage, router } from '@inertiajs/vue3';
import { computed } from 'vue';
import LineChart from '@/Components/LineChart.vue';

// 1. Get the props from Laravel
const props = defineProps({
    postureChunks: Array
});

// 2. [THE FIX] Safely get the user
// Instead of accessing it directly (which crashes if missing),
// we use a computed property with optional chaining (?.)
const page = usePage();
const user = computed(() => page.props.auth?.user || { name: 'User' });

// 3. Chart Data Logic
const chartData = computed(() => {
    // Handle empty data safely
    if (!props.postureChunks || props.postureChunks.length === 0) {
        return null;
    }

    const chunks = [...props.postureChunks].reverse();
    const labels = chunks.map(chunk =>
        new Date(chunk.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    );
    const scoreData = chunks.map(chunk => chunk.score);

    return {
        labels: labels,
        datasets: [
            {
                label: 'Posture Score',
                backgroundColor: '#3b82f6', // Blue-500
                borderColor: '#3b82f6',
                data: scoreData,
                tension: 0.3
            }
        ]
    };
});

// Test function (You can remove this later)
function testCreateChunk() {
    const fakeData = {
        score: Math.floor(Math.random() * 40 + 60),
        slouch_duration: Math.floor(Math.random() * 60),
        alert_count: Math.floor(Math.random() * 5)
    };

    router.post('/posture-chunks', fakeData, {
        preserveScroll: true,
        onSuccess: () => console.log("Test saved"),
        onError: (err) => console.error(err)
    });
}
</script>

<template>
    <Head title="Dashboard" />

    <AuthenticatedLayout>
        <template #header>
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">Analytics Dashboard</h2>
        </template>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8 space-y-6">

                <!-- Welcome Card -->
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 text-gray-900">
                        Welcome back, <strong>{{ user.name }}</strong>!
                    </div>
                </div>

                <!-- Chart Section -->
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="flex justify-between items-center mb-6">
                        <h3 class="text-lg font-semibold text-gray-700">Your Progress</h3>
                        <button
                            @click="testCreateChunk"
                            class="text-sm px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded text-gray-600 transition"
                        >
                            + Add Test Data
                        </button>
                    </div>

                    <!-- Chart Container -->
                    <div class="h-[400px] w-full">
                        <div v-if="!chartData" class="h-full flex items-center justify-center text-gray-400 bg-gray-50 rounded-lg border-2 border-dashed">
                            No data recorded yet. Start the camera to track your posture!
                        </div>
                        <LineChart v-else :chart-data="chartData" />
                    </div>
                </div>

                <!-- Recent Logs List -->
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Recent Logs</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm text-left text-gray-500">
                            <thead class="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th class="px-6 py-3">Time</th>
                                    <th class="px-6 py-3">Score</th>
                                    <th class="px-6 py-3">Slouch Time</th>
                                    <th class="px-6 py-3">Alerts</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="chunk in postureChunks" :key="chunk.id" class="bg-white border-b hover:bg-gray-50">
                                    <td class="px-6 py-4">{{ new Date(chunk.created_at).toLocaleString() }}</td>
                                    <td class="px-6 py-4 font-bold" :class="chunk.score >= 70 ? 'text-green-600' : 'text-red-600'">
                                        {{ chunk.score }}%
                                    </td>
                                    <td class="px-6 py-4">{{ chunk.slouch_duration }}s</td>
                                    <td class="px-6 py-4">{{ chunk.alert_count }}</td>
                                </tr>
                                <tr v-if="!postureChunks.length">
                                    <td colspan="4" class="px-6 py-4 text-center">No logs available.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    </AuthenticatedLayout>
</template>
