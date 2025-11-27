<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\PostureChunkController; // <-- Import this

// This line is usually there by default:
Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

// --- ADD THIS BLOCK ---
Route::middleware('auth:sanctum')->group(function () {
    // The route for your Python script
    Route::post('/posture-chunks', [PostureChunkController::class, 'store']);
});