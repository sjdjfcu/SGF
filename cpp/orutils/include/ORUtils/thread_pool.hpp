#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <atomic>
#include <string>
namespace tools{
    class TaskThreadPool{
    private:
        struct task_element_t {
            bool run_with_id;
            const std::function< void() > no_id;
            const std::function< void(std::size_t) > with_id;
            
            explicit task_element_t(const std::function< void() >& f) :
            run_with_id(false), no_id(f), with_id(nullptr) { }
            explicit task_element_t(const std::function< void(std::size_t) >& f) :
            run_with_id(true), no_id(nullptr), with_id(f) { }
        };
        std::queue<task_element_t> tasks_;
        std::vector<std::thread> threads_;
        std::mutex mutex_;
        std::condition_variable condition_;
        std::condition_variable completed_;
        bool running_;
        std::atomic_bool complete_;
        std::size_t available_;
        std::size_t total_;
        bool pause_;
    public:
        /// @brief Constructor.
        explicit TaskThreadPool(){
//            for ( std::size_t i = 0; i < 1; ++i ) {
//                threads_[i] = std::thread(std::bind(&TaskThreadPool::main_loop, this, i));
//            }
        };
        explicit TaskThreadPool(std::size_t pool_size)
        :  threads_(pool_size), running_(true), complete_(true),
        available_(pool_size), total_(pool_size), pause_(false) {
            for ( std::size_t i = 0; i < pool_size; ++i ) {
                threads_[i] = std::thread(std::bind(&TaskThreadPool::main_loop, this, i));
            }
        }
        void setPoolSize(size_t pool_size){
            threads_.clear();
            threads_.resize(pool_size);
            available_ = pool_size;
            total_ = pool_size;
            running_ = true;
            complete_ = true;
            pause_ = false;
            for ( std::size_t i = 0; i < pool_size; ++i ) {
                threads_[i] = std::thread(std::bind(&TaskThreadPool::main_loop, this, i));
            }
        }
        size_t getPoolSize(){return threads_.size();}
        bool finished(){return complete_;} 
        /// @brief Destructor.
        ~TaskThreadPool() {
            // Set running flag to false then notify all threads.
            {
                std::unique_lock< std::mutex > lock(mutex_);
                running_ = false;
                condition_.notify_all();
            }
            
            try {
                for (auto& t : threads_) {
                    t.join();
                }
            }
            // Suppress all exceptions.
            catch (const std::exception&) {}
        }
        
        /// @brief Add task to the thread pool if a thread is currently available.
        template <typename Task>
        void runTask(Task task) {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Set task and signal condition variable so that a worker thread will
            // wake up and use the task.
            tasks_.push(task_element_t(static_cast<std::function< void() >>(task)));
            complete_ = false;
            condition_.notify_one();
        }
        
        template <typename Task>
        void runTaskWithID(Task task) {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Set task and signal condition variable so that a worker thread will
            // wake up and use the task.
            tasks_.push(task_element_t(static_cast<std::function< void(std::size_t) >>(
                                                                                       task)));
            complete_ = false;
            condition_.notify_one();
        }
        
        /// @brief Wait for queue to be empty
        void waitWorkComplete() {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!complete_)
                completed_.wait(lock);
        }
        
        /**
         Wait for queue to be empty while showing the progress
         @param total_num the total size of input was passed into the pool
         @param display_time The interval of displaying estimation time left. (second)
         */
        void waitWorkCompleteandShow(size_t total_num, long long display_time = 1){
            float time_past = display_time;
            static std::string pbstr = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
            static int pbwidh = 60;
            while(checkTaskLeft()){
                std::this_thread::sleep_for(std::chrono::seconds(display_time));
                int current = checkTaskLeft();
                int proceed = total_num-current;
                float sec_left = time_past/proceed*current;
                time_past += display_time;
                //                printf("%5.2f%%[%d/%zu]ET: %4.0f(s) \n", float(proceed)/total_num*100, proceed, total_num, sec_left);
                
                float percentage = float(proceed)/total_num;
                int val = (int) (percentage * 100);
                int lpad = (int) (percentage * pbwidh);
                int rpad = pbwidh - lpad;
                printf ("\r[%.*s%*s] %3d%% ET: %4.0f(s)", lpad, pbstr.c_str(), rpad, "", val, sec_left);
                fflush (stdout);
            }
            printf("\n");
        }
        
        size_t checkTaskLeft(){return tasks_.size();}
        
        /** The pause funciton will wait for remaining job to complete and pause, until the function is called again with false value.  */
        void pause(bool option){
            pause_ = option;
            if(pause_){
                while(available_ != total_){
                    // wait...;
                }
            }
        }
        
    private:
        /// @brief Entry point for pool threads.
        void main_loop(std::size_t index) {
            while (running_) {
                while (pause_){
                    
                }
                
                // Wait on condition variable while the task is empty and
                // the pool is still running.
                std::unique_lock<std::mutex> lock(mutex_);
                while (tasks_.empty() && running_) {
                    condition_.wait(lock);
                }
                
                // If pool is no longer running, break out of loop.
                if (!running_) break;
                
                
                
                // Copy task locally and remove from the queue.  This is
                // done within its own scope so that the task object is
                // destructed immediately after running the task.  This is
                // useful in the event that the function contains
                // shared_ptr arguments bound via bind.
                {
                    auto tasks = tasks_.front();
                    tasks_.pop();
                    // Decrement count, indicating thread is no longer available.
                    --available_;
                    
                    lock.unlock();
                    
                    // Run the task.
                    try {
                        if (tasks.run_with_id) {
                            tasks.with_id(index);
                        } else {
                            tasks.no_id();
                        }
                    }
                    // Suppress all exceptions.
                    catch ( const std::exception& ) {}
                    
                    // Update status of empty, maybe
                    // Need to recover the lock first
                    lock.lock();
                    
                    // Increment count, indicating thread is available.
                    ++available_;
                    
                    
                    
                    if (tasks_.empty() && available_ == total_) {
                        complete_ = true;
                        completed_.notify_one();
                    }
                }
            }  // while running_
        }
    };
} // end of namespace tools
#endif

