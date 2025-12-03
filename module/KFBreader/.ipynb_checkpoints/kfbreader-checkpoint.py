import io

from PIL import Image
from openslide import AbstractSlide, _OpenSlideMap

import os
from ctypes import *
from itertools import count

from openslide import lowlevel

import numpy as np
from io import BytesIO
from PIL import Image

__all__ = ['KFBSlide']

_lib = cdll.LoadLibrary(os.path.join(os.path.split(__file__)[0], 'libkfbreader.so'))
_libimgop = os.path.join(os.path.split(__file__)[0], 'libImageOperationLib.so')

class KFBSlideError(Exception):
    """docstring for KFBSlideError"""

class _KfbSlide(object):
    def __init__(self, ptr):
        self._as_parameter_ = ptr
        self._valid = True
        self._close = kfbslide_close # 即使调用了close，资源也不能正确释放

    def __del__(self):
        if self._valid:
            self._close(self)

    def invalidate(self):
        self._valid = False

    @classmethod
    def from_param(cls, obj):
        if obj.__class__ != cls:
            raise ValueError("Not an KfbSlide reference")
        if not obj._as_parameter_:
            raise ValueError("Passing undefined slide object")
        if not obj._valid:
            raise ValueError("Passing closed kfbSlide object")
        return obj


# check for errors opening an image file and wrap the resulting handle
def _check_open(result, _func, _args):
    if result is None:
        raise lowlevel.OpenSlideUnsupportedFormatError(
            "Unsupported or missing image file")
    slide = _KfbSlide(c_void_p(result))
    # err = get_error(slide)
    # if err is not None:
    #     raise lowlevel.OpenSlideError(err)
    return slide


# prevent further operations on slide handle after it is closed
def _check_close(_result, _func, args):
    args[0].invalidate()


# check if the library got into an error state after each library call
def _check_error(result, func, args):
    # err = get_error(args[0])
    # if err is not None:
    #     raise lowlevel.OpenSlideError(err)
    return lowlevel._check_string(result, func, args)


# Convert returned NULL-terminated char** into a list of strings
def _check_name_list(result, func, args):
    _check_error(result, func, args)
    names = []
    for i in count():
        name = result[i]
        if not name:
            break
        names.append(name.decode('UTF-8', 'replace'))
    return names

# resolve and return an OpenSlide function with the specified properties
def _func(name, restype, argtypes, errcheck=_check_error):
    func = getattr(_lib, name)
    func.argtypes = argtypes
    func.restype = restype
    if errcheck is not None:
        func.errcheck = errcheck
    return func

detect_vendor = _func("_Z22kfbslide_detect_vendorPKc", c_char_p, [lowlevel._utf8_p],
                      lowlevel._check_string)

_kfbslide_open = _func("_Z13kfbslide_openPKcS0_", c_void_p, [lowlevel._utf8_p, lowlevel._utf8_p], _check_open)

kfbslide_buffer_free = _func("_Z20kfbslide_buffer_freeP9ImgHandlePh", c_bool, [c_void_p, POINTER(c_ubyte)])

kfbslide_close = _func("_Z14kfbslide_closeP9ImgHandle", None, [_KfbSlide], lowlevel._check_close)

kfbslide_get_level_count = _func("_Z24kfbslide_get_level_countP9ImgHandle", c_int32, [_KfbSlide])

_kfbslide_get_level_dimensions = _func("_Z29kfbslide_get_level_dimensionsP9ImgHandleiPxS1_", None,
                                       [_KfbSlide, c_int32, POINTER(c_int64), POINTER(c_int64)])


kfbslide_get_level_downsample = _func("_Z29kfbslide_get_level_downsampleP9ImgHandlei",
                                      c_double, [_KfbSlide, c_int32])

kfbslide_get_best_level_for_downsample = _func("_Z38kfbslide_get_best_level_for_downsampleP9ImgHandled", c_int32, [_KfbSlide, c_double])

_kfbslide_read_region = _func("_Z20kfbslide_read_regionP9ImgHandleiiiPiPPh", c_bool,
                              [_KfbSlide, c_int32, c_int32, c_int32, POINTER(c_int), POINTER(POINTER(c_ubyte))])

_kfbslide_read_roi_region = _func("_Z29kfbslide_get_image_roi_streamP9ImgHandleiiiiiPiPPh", c_bool,
                                  [_KfbSlide, c_int32, c_int32, c_int32, c_int32, c_int32, POINTER(c_int),
                                   POINTER(POINTER(c_ubyte))])

kfbslide_property_names = _func("_Z23kfbslide_property_namesP9ImgHandle", POINTER(c_char_p),
                                [_KfbSlide], _check_name_list)

kfbslide_property_value = _func("_Z23kfbslide_property_valueP9ImgHandlePKc", c_char_p,
                                [_KfbSlide, lowlevel._utf8_p])

_kfbslide_get_associated_image_names = _func("_Z35kfbslide_get_associated_image_namesP9ImgHandle", POINTER(c_char_p), [_KfbSlide],
                                             _check_name_list)

_kfbslide_get_associated_image_dimensions = _func("_Z40kfbslide_get_associated_image_dimensionsP9ImgHandlePKcPxS3_S3_", c_void_p,
                                                  [_KfbSlide, lowlevel._utf8_p, POINTER(c_int64), POINTER(c_int64),
                                                   POINTER(c_int)])
_kfbslide_read_associated_image = _func("_Z30kfbslide_read_associated_imageP9ImgHandlePKc", c_void_p,[_KfbSlide, lowlevel._utf8_p])


def kfbslide_open(name):
    osr = _kfbslide_open(_libimgop, name)
    return osr

def kfbslide_get_level_dimensions(osr, level):
    w, h = c_int64(), c_int64()
    _kfbslide_get_level_dimensions(osr, level, byref(w), byref(h))
    return (w.value, h.value)

def kfbslide_read_region(osr, level, pos_x, pos_y):
    data_length = c_int()
    pixel = POINTER(c_ubyte)()
    if not _kfbslide_read_region(osr, level, pos_x, pos_y, byref(data_length), byref(pixel)):
        raise ValueError("Fail to read region")
    import numpy as np
    img = Image.open(io.BytesIO(np.ctypeslib.as_array(pixel, shape=(data_length.value,))))
    kfbslide_buffer_free(osr, pixel)
    return img

def kfbslide_read_roi_region(osr, level, pos_x, pos_y, width, height):
    data_length = c_int()
    pixel = POINTER(c_ubyte)()
    if not _kfbslide_read_roi_region(osr, level, pos_x, pos_y, width, height, byref(data_length), byref(pixel)):
        raise ValueError("Fail to read roi region")
    import numpy as np
    img = Image.open(io.BytesIO(np.ctypeslib.as_array(pixel, shape=(data_length.value,))))
    kfbslide_buffer_free(osr, pixel)
    return img

def kfbslide_get_associated_image_names(osr):
    names = _kfbslide_get_associated_image_names(osr)
    rtn = []
    for name in names:
        if name is None:
            break
        rtn.append(name)
    return rtn

def kfbslide_get_associated_image_dimensions(osr, name):
    w, h = c_int64(), c_int64()
    data_length = c_int()
    _kfbslide_get_associated_image_dimensions(osr, name, byref(w), byref(h), byref(data_length))
    return (w.value, h.value), data_length.value

def kfbslide_read_associated_image(osr, name):
    data_length = kfbslide_get_associated_image_dimensions(osr, name)[1]
    pixel = cast(_kfbslide_read_associated_image(osr, name), POINTER(c_ubyte))
    narray = np.ctypeslib.as_array(pixel, shape=(data_length,))
    ret = Image.open(BytesIO(narray))
    print('aaa')
    kfbslide_buffer_free(osr, pixel)
    return ret


class KFBSlide(AbstractSlide):
    def __init__(self, filename):
        AbstractSlide.__init__(self)
        self.__filename = filename
        self._osr = kfbslide_open(filename)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.__filename)

    @classmethod
    def detect_format(cls, filename):
        return detect_vendor(filename)

    def close(self):
        kfbslide_close(self._osr)

    @property
    def level_count(self):
        return kfbslide_get_level_count(self._osr)

    @property
    def level_dimensions(self):
        return tuple(kfbslide_get_level_dimensions(self._osr, i)
                     for i in range(self.level_count))

    @property
    def level_downsamples(self):
        return tuple(kfbslide_get_level_downsample(self._osr, i)
                     for i in range(self.level_count))

    @property
    def properties(self):
        return _KfbPropertyMap(self._osr)

    @property
    def associated_images(self):
        return _AssociatedImageMap(self._osr)

    def get_best_level_for_downsample(self, downsample):
        return kfbslide_get_best_level_for_downsample(self._osr, downsample)

    def read_region(self, location, level, size):
        x = int(location[0])
        y = int(location[1])
        width = int(size[0])
        height = int(size[1])

        tw, th = kfbslide_get_level_dimensions(self._osr, level)
        cw = min(tw - x, width)
        ch = min(th - y, height)
        try:
            import numpy as np
            img = kfbslide_read_roi_region(self._osr, level, x, y, cw, ch)
            return np.pad(img, ((0, height - ch), (0, width - cw), (0, 0)), constant_values=255)
        except:
            print(f'crop at ({x}, {y}, {cw}, {ch}) on ({tw},{th})')
            return np.ones((height, width, 3), dtype=np.uint8) * 255

    def get_thumbnail(self, size):
        """Return a PIL.Image containing an RGB thumbnail of the image.

        size:     the maximum size of the thumbnail."""
        return self.associated_images[b'thumbnail']


class _KfbPropertyMap(_OpenSlideMap):
    def _keys(self):
        return kfbslide_property_names(self._osr)

    def __getitem__(self, key):
        v = kfbslide_property_value(self._osr, key)
        if v is None:
            raise KeyError()
        return v


class _AssociatedImageMap(_OpenSlideMap):
    def _keys(self):
        return kfbslide_get_associated_image_names(self._osr)

    def __getitem__(self, key):
        if key not in self._keys():
            raise KeyError()
        return kfbslide_read_associated_image(self._osr, key)


    