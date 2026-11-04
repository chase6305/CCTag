/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cctag
{

class TagSemaphore
{
public:
    explicit TagSemaphore(int count) : _count(count) {}

    void wait(int count = 1);
    void post(int count = 1);

private:
    int _count;
    std::mutex _mutex;
    std::condition_variable _available;
};

class TagFrameGate
{
public:
    // Generation zero signals shutdown; frames start at generation one.
    std::size_t wait(std::size_t previous)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _ready.wait(lock, [&] { return _stopped || _generation != previous; });
        return _stopped ? 0 : _generation;
    }

    void post()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if(_stopped)
                return;
            ++_generation;
        }
        _ready.notify_all();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopped = true;
        }
        _ready.notify_all();
    }

    // Reset only after all workers from the previous run have joined.
    void reset()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _generation = 0;
        _stopped = false;
    }

private:
    std::size_t _generation = 0;
    bool _stopped = false;
    std::mutex _mutex;
    std::condition_variable _ready;
};

class TagThreads
{
public:
    // Call control methods from the pipeline owner, never from a layer callback.
    TagThreads() = default;
    ~TagThreads();

    // Replace the workers; failed thread creation leaves the run stopped.
    void init(int layers, std::function<void(int)> handleLayer);

    // Each accepted round finishes every layer before returning or rethrowing.
    void oneRound();

    // Finish the current round and join workers. Safe to repeat; new rounds need init().
    void stop();

private:
    void workerLoop(int layer);
    void stopWorkers(); // Caller holds _roundMutex.

    std::function<void(int)> _handleLayer;
    std::vector<std::thread> _workers;
    TagFrameGate _frameReady;
    TagSemaphore _frameDone{0};
    std::mutex _roundMutex;
    std::mutex _errorMutex;
    std::exception_ptr _failure;
};

} // namespace cctag
