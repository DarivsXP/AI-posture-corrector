<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class PostureChunk extends Model
{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = [
        'score',
        'slouch_duration',
        'alert_count',
        // 'user_id' is handled by the relationship,
        // so we don't need to add it here.
    ];

    /**
     * Get the user that owns the posture chunk.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
