"""
# Pyqtree

Pyqtree is a pure Python spatial index for GIS or rendering usage.
It stores and quickly retrieves items from a 2x2 rectangular grid area,
and grows in depth and detail as more items are added.
The actual quad tree implementation is adapted from
[Matt Rasmussen's compbio library](https://github.com/mdrasmus/compbio/blob/master/rasmus/quadtree.py)
and extended for geospatial use.


## Platforms

Python 2 and 3.


## Dependencies

Pyqtree is written in pure Python and has no dependencies.


## Installing It

Installing Pyqtree can be done by opening your terminal or commandline and typing:

    pip install pyqtree

Alternatively, you can simply download the "pyqtree.py" file and place
it anywhere Python can import it, such as the Python site-packages folder.


## Example Usage

Start your script by importing the quad tree.

    from pyqtree import Index

Setup the spatial index, giving it a bounding box area to keep track of.
The bounding box being in a four-tuple: (xmin, ymin, xmax, ymax).

    spindex = Index(bbox=(0, 0, 100, 100))

Populate the index with items that you want to be retrieved at a later point,
along with each item's geographic bbox.

    # this example assumes you have a list of items with bbox attribute
    for item in items:
        spindex.insert(item, item.bbox)

Then when you have a region of interest and you wish to retrieve items from that region,
just use the index's intersect method. This quickly gives you a list of the stored items
whose bboxes intersects your region of interests.

    overlapbbox = (51, 51, 86, 86)
    matches = spindex.intersect(overlapbbox)

There are other things that can be done as well, but that's it for the main usage!


## More Information:

- [Home Page](http://github.com/karimbahgat/Pyqtree)
- [API Documentation](https://karimbahgat.github.io/Pyqtree/)


## License:

This code is free to share, use, reuse, and modify according to the MIT license:

The MIT License (MIT)

Copyright (c) 2018 Karim Bahgat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


## Credits:

- Karim Bahgat
- Joschua Gandert

"""

__version__ = "1.0.0"

#PYTHON VERSION CHECK
import sys
PYTHON3 = int(sys.version[0]) == 3
if PYTHON3:
    xrange = range


def _normalize_rect(rect):
    if len(rect) == 2:
        x1, y1 = rect
        x2, y2 = rect
    else:
        x1, y1, x2, y2 = rect
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _loopallchildren(parent):
    for child in parent.children:
        if child.children:
            for subchild in _loopallchildren(child):
                yield subchild
        yield child


class _QuadNode(object):
    def __init__(self, item, rect):
        self.item = item
        self.rect = rect

    def __eq__(self, other):
        return self.item == other.item and self.rect == other.rect

    def __hash__(self):
        return hash(self.item)


class _QuadTree(object):
    """
    Internal backend version of the index.
    The index being used behind the scenes. Has all the same methods as the user
    index, but requires more technical arguments when initiating it than the
    user-friendly version.
    """

    def __init__(self, x, y, width, height, max_items, max_depth, _depth=0):
        self.nodes = []
        self.children = []
        self.center = (x, y)
        self.width, self.height = width, height
        self.max_items = max_items
        self.max_depth = max_depth
        self._depth = _depth

    def __iter__(self):
        for child in _loopallchildren(self):
            yield child

    def _insert(self, item, bbox):
        rect = _normalize_rect(bbox)
        if len(self.children) == 0:
            node = _QuadNode(item, rect)
            self.nodes.append(node)

            if len(self.nodes) > self.max_items and self._depth < self.max_depth:
                self._split()
        else:
            self._insert_into_children(item, rect)

    def _remove(self, item, bbox):
        rect = _normalize_rect(bbox)
        if len(self.children) == 0:
            node = _QuadNode(item, rect)
            self.nodes.remove(node)
        else:
            self._remove_from_children(item, rect)

    def _intersect(self, rect, results=None, uniq=None):
        if results is None:
            rect = _normalize_rect(rect)
            results = []
            uniq = set()
        # search children
        if self.children:
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[0]._intersect(rect, results, uniq)
                if rect[3] >= self.center[1]:
                    self.children[1]._intersect(rect, results, uniq)
            if rect[2] >= self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[2]._intersect(rect, results, uniq)
                if rect[3] >= self.center[1]:
                    self.children[3]._intersect(rect, results, uniq)
        # search node at this level
        for node in self.nodes:
            _id = id(node.item)
            if (_id not in uniq and
                node.rect[2] >= rect[0] and node.rect[0] <= rect[2] and
                node.rect[3] >= rect[1] and node.rect[1] <= rect[3]):
                results.append(node.item)
                uniq.add(_id)
        return results

    def _insert_into_children(self, item, rect):
        # if rect spans center then insert here
        if (rect[0] <= self.center[0] and rect[2] >= self.center[0] and
            rect[1] <= self.center[1] and rect[3] >= self.center[1]):
            node = _QuadNode(item, rect)
            self.nodes.append(node)
        else:
            # try to insert into children
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[0]._insert(item, rect)
                if rect[3] >= self.center[1]:
                    self.children[1]._insert(item, rect)
            if rect[2] > self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[2]._insert(item, rect)
                if rect[3] >= self.center[1]:
                    self.children[3]._insert(item, rect)

    def _remove_from_children(self, item, rect):
        # if rect spans center then insert here
        if (rect[0] <= self.center[0] and rect[2] >= self.center[0] and
            rect[1] <= self.center[1] and rect[3] >= self.center[1]):
            node = _QuadNode(item, rect)
            self.nodes.remove(node)
        else:
            # try to remove from children
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[0]._remove(item, rect)
                if rect[3] >= self.center[1]:
                    self.children[1]._remove(item, rect)
            if rect[2] > self.center[0]:
                if rect[1] <= self.center[1]:
                    self.children[2]._remove(item, rect)
                if rect[3] >= self.center[1]:
                    self.children[3]._remove(item, rect)

    def _split(self):
        quartwidth = self.width / 4.0
        quartheight = self.height / 4.0
        halfwidth = self.width / 2.0
        halfheight = self.height / 2.0
        x1 = self.center[0] - quartwidth
        x2 = self.center[0] + quartwidth
        y1 = self.center[1] - quartheight
        y2 = self.center[1] + quartheight
        new_depth = self._depth + 1
        self.children = [_QuadTree(x1, y1, halfwidth, halfheight,
                                   self.max_items, self.max_depth, new_depth),
                         _QuadTree(x1, y2, halfwidth, halfheight,
                                   self.max_items, self.max_depth, new_depth),
                         _QuadTree(x2, y1, halfwidth, halfheight,
                                   self.max_items, self.max_depth, new_depth),
                         _QuadTree(x2, y2, halfwidth, halfheight,
                                   self.max_items, self.max_depth, new_depth)]
        nodes = self.nodes
        self.nodes = []
        for node in nodes:
            self._insert_into_children(node.item, node.rect)

    def __len__(self):
        """
        Returns:

        - A count of the total number of members/items/nodes inserted
        into this quadtree and all of its child trees.
        """
        size = 0
        for child in self.children:
            size += len(child)
        size += len(self.nodes)
        return size


MAX_ITEMS = 10
MAX_DEPTH = 20


class Index(_QuadTree):
    """
    The top spatial index to be created by the user. Once created it can be
    populated with geographically placed members that can later be tested for
    intersection with a user inputted geographic bounding box. Note that the
    index can be iterated through in a for-statement, which loops through all
    all the quad instances and lets you access their properties.

    Example usage:

    >>> spindex = Index(bbox=(0, 0, 100, 100))
    >>> spindex.insert('duck', (50, 30, 53, 60))
    >>> spindex.insert('cookie', (10, 20, 15, 25))
    >>> spindex.insert('python', (40, 50, 95, 90))
    >>> results = spindex.intersect((51, 51, 86, 86))
    >>> sorted(results)
    ['duck', 'python']
    """

    def __init__(self, bbox=None, x=None, y=None, width=None, height=None, max_items=MAX_ITEMS, max_depth=MAX_DEPTH):
        """
        Initiate by specifying either 1) a bbox to keep track of, or 2) with an xy centerpoint and a width and height.

        Parameters:
        - **bbox**: The coordinate system bounding box of the area that the quadtree should
            keep track of, as a 4-length sequence (xmin,ymin,xmax,ymax)
        - **x**:
            The x center coordinate of the area that the quadtree should keep track of.
        - **y**
            The y center coordinate of the area that the quadtree should keep track of.
        - **width**:
            How far from the xcenter that the quadtree should look when keeping track.
        - **height**:
            How far from the ycenter that the quadtree should look when keeping track
        - **max_items** (optional): The maximum number of items allowed per quad before splitting
            up into four new subquads. Default is 10.
        - **max_depth** (optional): The maximum levels of nested subquads, after which no more splitting
            occurs and the bottommost quad nodes may grow indefinately. Default is 20.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width, height = abs(x2-x1), abs(y2-y1)
            midx, midy = x1+width/2.0, y1+height/2.0
            super(Index, self).__init__(midx, midy, width, height, max_items, max_depth)

        elif None not in (x, y, width, height):
            super(Index, self).__init__(x, y, width, height, max_items, max_depth)

        else:
            raise Exception("Either the bbox argument must be set, or the x, y, width, and height arguments must be set")

    def insert(self, item, bbox):
        """
        Inserts an item into the quadtree along with its bounding box.

        Parameters:
        - **item**: The item to insert into the index, which will be returned by the intersection method
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)
        """
        self._insert(item, bbox)

    def remove(self, item, bbox):
        """
        Removes an item from the quadtree.

        Parameters:
        - **item**: The item to remove from the index
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)

        Both parameters need to exactly match the parameters provided to the insert method.
        """
        self._remove(item, bbox)

    def intersect(self, bbox):
        """
        Intersects an input boundingbox rectangle with all of the items
        contained in the quadtree.

        Parameters:
        - **bbox**: A spatial bounding box tuple with four members (xmin,ymin,xmax,ymax)

        Returns:
        - A list of inserted items whose bounding boxes intersect with the input bbox.
        """
        return self._intersect(bbox)
