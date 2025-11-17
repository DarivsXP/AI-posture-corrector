<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class PostureChunk extends Model
{
    use HasFactory;

    // Add this function:
    /**
     * Get the user that owns the posture chunk.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
