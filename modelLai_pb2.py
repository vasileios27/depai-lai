# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelLai.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0emodelLai.proto\x12\x06model1\"/\n\tImageData\x12\x12\n\nimage_path\x18\x01 \x01(\t\x12\x0e\n\x06offset\x18\x02 \x01(\x05\"1\n\x0cImageRequest\x12!\n\x06images\x18\x01 \x03(\x0b\x32\x11.model1.ImageData\"\x94\x01\n\rImageResponse\x12\x35\n\x07results\x18\x01 \x03(\x0b\x32$.model1.ImageResponse.ProcessedImage\x1aL\n\x0eProcessedImage\x12\x12\n\nimage_path\x18\x01 \x01(\t\x12\x11\n\tprocessed\x18\x02 \x01(\x08\x12\x13\n\x0bresult_path\x18\x03 \x01(\t2M\n\x0eImageProcessor\x12;\n\x0cProcessImage\x12\x14.model1.ImageRequest\x1a\x15.model1.ImageResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'modelLai_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_IMAGEDATA']._serialized_start=26
  _globals['_IMAGEDATA']._serialized_end=73
  _globals['_IMAGEREQUEST']._serialized_start=75
  _globals['_IMAGEREQUEST']._serialized_end=124
  _globals['_IMAGERESPONSE']._serialized_start=127
  _globals['_IMAGERESPONSE']._serialized_end=275
  _globals['_IMAGERESPONSE_PROCESSEDIMAGE']._serialized_start=199
  _globals['_IMAGERESPONSE_PROCESSEDIMAGE']._serialized_end=275
  _globals['_IMAGEPROCESSOR']._serialized_start=277
  _globals['_IMAGEPROCESSOR']._serialized_end=354
# @@protoc_insertion_point(module_scope)
