<template>
  <h1>This is my secure posture app!</h1>
  <p>Welcome, {{ user.name }}</p>
</template>

<script setup>
import { usePage } from '@inertiajs/vue3';

// Get the logged-in user's data
const user = usePage().props.auth.user;
</script>
