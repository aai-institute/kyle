import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pprint import pprint
from typing import TypeVar, Generic, Dict, Any, Iterator, Tuple, Type, Union, List, Pattern

from kale.constants import _NAME_TOKEN, _METADATA_TOKEN
from kale.util import getFirstDuplicate, updateDictByListingValuesAndNormalize

log = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")


# Unfortunately, making a generic abstract Mapping type work properly currently seems to be impossible in intellij/python
# therefore we mimic the mapping type ourselves by using plain Generic
class HashableStruct(Generic[T], ABC):
    """
    Base class for hashable mappings of strings to objects of a determined type
    """
    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.toJSON() == other.toJSON()
        return False

    def __hash__(self):
        return hash(self.toJSON())

    @abstractmethod
    def toDict(self) -> Dict[str, Any]:
        """
        This method is primarily used for serializing hashable structs. It should return a mapping
        from strings to either objects of the same type as the struct's values or to the struct's metadata

        :return: a dict representation of the struct object
        """
        pass

    @abstractmethod
    def __getitem__(self, item: str) -> T:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass

    def keys(self):
        return self.__iter__()

    def values(self) -> Iterator[T]:
        return (self[k] for k in self.keys())

    def items(self) -> Iterator[Tuple[str, T]]:
        return zip(self.keys(), self.values())

    def toJSON(self):
        """
        :return: the serialized object as json string
        """
        return json.dumps(self, default=lambda o: o.toDict(), sort_keys=True)

    def saveAsJSON(self, path: str):
        """
        persist the JSON serialization as file

        :param path: file-path that will contain the JSON serialization
        """
        os.makedirs(os.path.abspath(os.path.dirname(path)), exist_ok=True)
        log.info(f"Saving {self.__class__.__name__} to {path}")
        with open(path, 'w') as f:
            json.dump(self, f, default=lambda o: o.toDict(), sort_keys=True)

    def show(self):
        """
        pretty print the JSON serialization
        """
        pprint(json.loads(self.toJSON()))


class NamedStruct(HashableStruct[T], ABC):
    """
    Base class for hashable structs that are named and contain metadata.
    """
    def __init__(self, name: str, metadata: Dict[str, Any] = None):
        self.name = name
        self.metadata = metadata if metadata is not None else {}

    @classmethod
    @abstractmethod
    def _instantiateFromDict(cls: Type[T], name: str, nameValuesDict: Dict[str, Union[str, Dict]]) -> T:
        pass

    @abstractmethod
    def _toDict(self: T) -> Dict[str, T]:
        """
        This should return a mapping entryName -> entry for all entries of the named struct
        (excluding the obligatory entries "name" and "metadata")
        """
        pass

    def toDict(self) -> Dict[str, Any]:
        d = self._toDict().copy()
        d[_NAME_TOKEN] = self.name
        d[_METADATA_TOKEN] = self.metadata
        return d

    @classmethod
    def fromJSON(cls: Type[S], pathOrDict: Union[str, dict]) -> S:
        """
        Deserialize a saved json to an an instance of the given class

        :param pathOrDict: path to a json file or dict (the latter would typically be a result of json.load)
        :return:
        """
        if isinstance(pathOrDict, dict):
            d = pathOrDict
        else:
            with open(pathOrDict, 'rb') as f:
                d = json.load(f)
        metadata = d.pop(_METADATA_TOKEN)
        name = str(d.pop(_NAME_TOKEN))
        assert isinstance(metadata, dict), f"Cannot deserialise json: invalid metadata entry: {metadata}"
        instance = cls._instantiateFromDict(name, d)
        instance.metadata = metadata
        return instance

    def __iter__(self):
        return self._toDict().__iter__()

    def __getitem__(self, item: str):
        return self._toDict()[item]


NamedStructType = TypeVar("NamedStructType", bound=NamedStruct)


# TODO: in python 3.8 it is possible to access generic types at runtime. Therefore the collection no longer
#  needs to be abstract as we can implement _instantiateFromDict along the lines of
#     @classmethod
#     def _instantiateFromDict(cls: T[NamedStructType], name: str, nameValuesDict: Dict[str, Union[str, Dict]]) -> T:
#         namedStructs = [NamedStructType.fromJSON(v) for v in nameValuesDict.values()]
#         return cls(name, *namedStructs)
class NamedStructCollection(NamedStruct[NamedStructType], ABC):
    """
    collection of :class:`NamedStruct` which itself has a name
    """

    def __init__(self, name: str, *namedStructs: NamedStruct, metadata: Dict[str, Any] = None):
        super().__init__(name, metadata=metadata)
        self._nameToNamedStruct = self._dictFromNamedStructs(*namedStructs)

    def _toDict(self):
        return self._nameToNamedStruct

    def drop(self, structNames: Union[str, List[str]]) -> __qualname__:
        """
        remove :class:`NamedStruct` by their name from the collection

        :param structNames: list of :class:`NamedStruct` to drop
        :return: :class:`NamedStructCollection`
        """
        if isinstance(structNames, str):
            structNames = [structNames]
        missingStructNames = set(structNames).difference(set(self.keys()))
        if missingStructNames:
            raise ValueError(f"Cannot drop entries that were not present: {missingStructNames}")
        for structName in structNames:
            self._nameToNamedStruct.pop(structName)
        return self.__class__(self.name, *list(self.values()), metadata=self.metadata)

    def getSubset(self, structNames: List[Union[str, Pattern]]) -> __qualname__:
        """
        create a :class:`NamedStructCollection` that only contains the specified :class:`NamedStruct`

        :param structNames: list of :class:`NamedStruct` that you want to keep.
            The names can also be provided as regular expression (e.g. ``re.compile("pattern.*")``), which will keep all :class:`NamedStruct` that match this pattern.
        :return: :class:`NamedStructCollection`
        """
        if not isinstance(structNames, list):
            structNames = [structNames]
        structNames = structNames

        namedStructsToKeep = []
        for structName in structNames:
            if isinstance(structName, Pattern):
                for name in self.keys():
                    if re.match(structName, name):
                        namedStructsToKeep.append(self._nameToNamedStruct[name])
            else:
                namedStruct = self._nameToNamedStruct.get(structName)
                if namedStruct:
                    namedStructsToKeep.append(namedStruct)
        return self.__class__(self.name, *namedStructsToKeep, metadata=self.metadata)

    @staticmethod
    def _dictFromNamedStructs(*namedStructs: NamedStruct) -> Dict[str, NamedStruct]:
        duplicateName = getFirstDuplicate([s.name for s in namedStructs])
        if duplicateName is not None:
            raise ValueError(f"Found duplicate name in input sequence: {duplicateName}")

        result = {}
        for struct in namedStructs:
            if struct.name in [_METADATA_TOKEN, _NAME_TOKEN]:
                raise ValueError(f"Forbidden struct name: {struct.name}")
            result[struct.name] = struct
        return result

    def merge(self, *namedStructCollections: 'NamedStructCollection'):
        """
        Merges the current struct collection with the given ones. The merge strategy is the following:

           1) This merge only adds new named structs to the collection, thus it is of the type 'discard'.

           2) Metadata dictionaries are merged through the inbuilt update method (i.e. merge of type 'replace')

           3) The set of the structCollections' names is merged together to a single string

        :param namedStructCollections: :class:`NamedStructCollection`
        :return: instance of the the current object's class
        """
        if not namedStructCollections:
            return self
        namedStructCollections = (self, ) + namedStructCollections

        metaDataDicts = []
        mergedStructs = set()
        mergedStructNames = set()
        mergedNames = set()
        for namedStructCollection in namedStructCollections:
            if namedStructCollection.__class__ != self.__class__:
                raise Exception(f"cannot merge with object of incompatible class: {namedStructCollection.__class__.__name__}")
            metaDataDicts.append(namedStructCollection.metadata)
            newNamedStructs = {struct for struct in namedStructCollection.values() if struct.name not in mergedStructNames}

            mergedStructs = mergedStructs.union(newNamedStructs)
            mergedStructNames = mergedStructNames.union({struct.name for struct in newNamedStructs})
            mergedNames.add(namedStructCollection.name)
        if len(mergedNames) == 1:
            mergedName = self.name
        else:
            mergedName = f"merge_of:[{','.join(sorted(mergedNames))}]"
        result = self.__class__(mergedName, *mergedStructs)

        result.metadata = updateDictByListingValuesAndNormalize(metaDataDicts)
        return result