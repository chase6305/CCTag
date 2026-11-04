/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "tag_threads.h"

#include <stdexcept>
#include <utility>

namespace cctag
{

TagThreads::~TagThreads()
{
    stop();
}

void TagThreads::init(int layers, std::function<void(int)> handleLayer)
{
    if(layers <= 0 || !handleLayer)
        throw std::invalid_argument("TagThreads requires layers and a frame handler");

    std::lock_guard<std::mutex> roundLock(_roundMutex);
    stopWorkers();
    _handleLayer = std::move(handleLayer);
    _frameReady.reset();
    try
    {
        _workers.reserve(layers);
        for(int layer = 0; layer < layers; ++layer)
            _workers.emplace_back(&TagThreads::workerLoop, this, layer);
    }
    catch(...)
    {
        // Thread creation can fail after earlier workers have started waiting.
        stopWorkers();
        throw;
    }
}

void TagThreads::oneRound()
{
    std::lock_guard<std::mutex> roundLock(_roundMutex);
    if(_workers.empty())
        throw std::logic_error("TagThreads has not been initialized or has stopped");

    _frameReady.post();
    _frameDone.wait(static_cast<int>(_workers.size()));

    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> errorLock(_errorMutex);
        failure = std::exchange(_failure, nullptr);
    }
    if(failure)
        std::rethrow_exception(failure);
}

void TagThreads::stop()
{
    std::lock_guard<std::mutex> roundLock(_roundMutex);
    stopWorkers();
}

void TagThreads::stopWorkers()
{
    _frameReady.stop();
    for(auto& worker : _workers)
        worker.join();
    _workers.clear();
    _handleLayer = nullptr;
    _failure = nullptr;
}

void TagThreads::workerLoop(int layer)
{
    std::size_t generation = 0;
    while((generation = _frameReady.wait(generation)) != 0)
    {
        try
        {
            _handleLayer(layer);
        }
        catch(...)
        {
            std::lock_guard<std::mutex> errorLock(_errorMutex);
            if(!_failure)
                _failure = std::current_exception();
        }
        _frameDone.post();
    }
}

void TagSemaphore::wait(int count)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _available.wait(lock, [&] { return _count >= count; });
    _count -= count;
}

void TagSemaphore::post(int count)
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _count += count;
    }
    _available.notify_all();
}

} // namespace cctag
