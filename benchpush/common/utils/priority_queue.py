import queue


class PriorityQueue(queue.PriorityQueue):
    """
    Slightly modified priority queue (lowest first) with an update
    function to update items with a lower priority. Items are of
    the form (priority number, data)
    """
    def __init__(self, item=None):
        self.invalidate = set()  # hash table to keep track of items invalidated during an update
        super().__init__()

        if item:
            self.put(item)

    def get_item(self):
        """
        Custom getter which continuously pops items off
        the queue until a valid one is found
        """
        while True:
            item = self.get(block=False)
            if item in self.invalidate:
                continue
            return item

    def update(self, orig_item, new_item):
        """
        Invalidate original item and add new item to min priority queue
        """
        self.invalidate.add(orig_item)  # mark item with old/lower priority as invalid
        self.put(new_item)
