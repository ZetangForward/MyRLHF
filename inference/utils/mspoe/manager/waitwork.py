import os,sys,random
from typing import List,Optional
import torch
from argparse import Namespace
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor



class ProducerConsumerManager:
    def __init__(self,
                 task_info_list : list=[],
                 max_producers: int = 1,
                 max_consumers: int = 1,
                 produce_config: Namespace=Namespace(),
                 consume_config: Namespace=Namespace(),
                 produce_env_config: Optional[Namespace]=None,
                 consume_env_config: Optional[Namespace]=None
                 ):
        self.task_info_list=[_ for _ in task_info_list]
        random.shuffle(self.task_info_list)
        self.max_producers=max_producers
        self.max_consumers=max_consumers
        self.produce_config=produce_config
        self.consume_config=consume_config
        self.produce_env_config=produce_env_config
        self.consume_env_config=consume_env_config

        self.producers=[]
        self.consumers=[]

        mp.get_start_method("spawn")

        self._task_queue=mp.Queue()
        self.manager=mp.Manager()
        self._result_list=self.manager.list()

    
    def __exit__(self,exc_type, exc_value, traceback):
        self.manager.shutdown()
        for process in self.producers+self.consumers:
            process.terminate()
        return False
    def __enter__(self):
        for i in range(self.max_consumers):
            self.set_consumer_environment(i,self.consume_env_config)
            process=mp.Process(
                target=self._consumer,
                args=(self._task_queue,self.consume_config, self._result_list)
            )
            self.consumers.append(process)
            process.start()
            

        for i,task_info_list in enumerate(self.chunks(self.task_info_list,self.max_producers)):
            self.set_procuder_environment(i,self.produce_env_config)
            process=mp.Process(
                target=self._producer,
                args=(task_info_list, self._task_queue, self.produce_config)
            )
            
            self.producers.append(process)
            process.start()

        
        assert len(self.producers)==self.max_producers, "Chunk Error!!"


        for process in self.producers:
            process.join()
        
        print("Process Over!")

        for _ in range(self.max_consumers):
            self._task_queue.put(None)
    
        for process in self.consumers:
            process.join()        

        self.result_list=list(self._result_list)

        return self
    
    @classmethod
    def chunks(cls, lst, chunk_num):
        chunk_size=(len(lst)-1)//chunk_num + 1
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    @classmethod
    def set_procuder_environment(cls, producer_id: int, produce_env_config: Namespace):
        '''
        set the environment variable for producers before created

        produce_env_config is the object you passed when initating

        '''
        if produce_env_config is not None:raise NotImplementedError
    @classmethod
    def set_consumer_environment(cls, consumer_id: int, consume_env_config: Namespace):
        '''
        
        set the environment variable for consumers before created

        consume_env_config is the object you passed when initating
        '''
        
        if consume_env_config is not None:raise NotImplementedError

    @classmethod
    def _producer(cls, task_info_list: list, task_queue: mp.Queue, produce_config: Namespace):
        cls.set_producer_global_variable(produce_config)
        for task_info in tqdm(task_info_list,desc=f"Producer: {os.getpid()}"):
            for task_sample in tqdm(cls.produce(task_info, produce_config),desc=f"P single: {os.getpid()}"):
                task_queue.put(task_sample)
    @classmethod
    def _consumer(cls, task_queue: mp.Queue, consume_config: Namespace, result_list: List):
        progress_bar = tqdm(desc=f"Consumer: {os.getpid()}", total=0)  
        delta=2
        number_tasks=0
        progress_bar.total=delta
        
        cls.set_consumer_global_variable(consume_config)
        while True:
            task_sample=task_queue.get()
            if task_sample is None:break
            for task_result in tqdm(cls.consume(task_sample, consume_config),desc=f"C single: {os.getpid()}", total=0):
                result_list.append(task_result)
            
            number_tasks+=1
            if number_tasks==progress_bar.total:
                progress_bar.total*=delta
            progress_bar.update(1)
        progress_bar.total=number_tasks
        progress_bar.close()

    @classmethod
    def set_producer_global_variable(cls, produce_config: Namespace):
        '''
        set the public global in every producer process, which will be used during loop
        '''
        pass
    @classmethod
    def set_consumer_global_variable(cls, consume_config: Namespace):
        '''
        set the public global in every consumer process, which will be used during loop
        '''
        pass

    @classmethod
    def produce(cls, task_info , produce_config: Namespace) -> object:
        '''
        task_info is the sample in task_info_list you passed when initiating the manager
        produce_config is also the object you passed when initiating.

        After gaining a sample , do : yield sample
        
        '''
        raise NotImplementedError

    @classmethod
    def consume(cls, task_sample, consume_config: Namespace) -> object:
        '''
        task_sample is the return value of the   `produce` you implement
        consume_config is the object you passed when initating

        After gaining a result, do : yield result
        '''
        raise NotImplementedError


if __name__=="__main__":
    with ProducerConsumerManager(
        task_info_list=[{'task_type':'x','info': 'int'},{'task_type':'y','info': 'str'}],
        max_producers=2,
        max_consumers=2,
        produce_config=Namespace(),
        consume_config=Namespace()
    ) as manager:
        result_list=manager.result_list
