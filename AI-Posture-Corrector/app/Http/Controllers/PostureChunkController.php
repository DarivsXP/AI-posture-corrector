<?php

namespace App\Http\Controllers;

use App\Models\PostureChunk;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth; // <--- 1. Import this
use Inertia\Inertia;

class PostureChunkController extends Controller
{
    public function index()
    {
        /** @var \App\Models\User $user */ //
        $user = Auth::user();

        $chunks = $user->postureChunks()->orderBy('created_at', 'desc')->get();

        return Inertia::render('Dashboard', [
            'postureChunks' => $chunks
        ]);
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'score'           => 'required|integer|min:0|max:100',
            'slouch_duration' => 'required|integer|min:0',
            'alert_count'     => 'required|integer|min:0',  
        ]);

        // FIX: Use $request->user() instead of auth()->user()
        // Since you already have the $request variable, this is cleaner.
        $chunk = $request->user()->postureChunks()->create($validated);

        if ($request->wantsJson()) {
            return response()->json([
                'message' => 'Chunk saved successfully!',
                'chunk' => $chunk
            ], 201);
        }

        return redirect()->back()->with('message', 'Chunk saved successfully!');
    }
}