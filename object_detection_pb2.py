# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16object_detection.proto\x12\x10object_detection\"Q\n\x08Response\x12(\n\x06\x62\x62oxes\x18\x01 \x01(\x0b\x32\x18.object_detection.BBoxes\x12\x0e\n\x06signal\x18\x02 \x01(\x05\x12\x0b\n\x03\x66ps\x18\x03 \x01(\x05\"=\n\x07Request\x12\x12\n\nframe_data\x18\x01 \x01(\x0c\x12\x0b\n\x03\x66ps\x18\x02 \x01(\x05\x12\x11\n\tclient_id\x18\x03 \x01(\x05\"\x16\n\x06\x42\x42oxes\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"\r\n\x0bInitRequest\"!\n\x0cInitResponse\x12\x11\n\tclient_id\x18\x01 \x01(\x05\"!\n\x0c\x43loseRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\"\x0f\n\rCloseResponse2\xf4\x01\n\x08\x44\x65tector\x12\x41\n\x06\x64\x65tect\x12\x19.object_detection.Request\x1a\x1a.object_detection.Response\"\x00\x12N\n\x0binit_client\x12\x1d.object_detection.InitRequest\x1a\x1e.object_detection.InitResponse\"\x00\x12U\n\x10\x63lose_connection\x12\x1e.object_detection.CloseRequest\x1a\x1f.object_detection.CloseResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_RESPONSE']._serialized_start=44
  _globals['_RESPONSE']._serialized_end=125
  _globals['_REQUEST']._serialized_start=127
  _globals['_REQUEST']._serialized_end=188
  _globals['_BBOXES']._serialized_start=190
  _globals['_BBOXES']._serialized_end=212
  _globals['_INITREQUEST']._serialized_start=214
  _globals['_INITREQUEST']._serialized_end=227
  _globals['_INITRESPONSE']._serialized_start=229
  _globals['_INITRESPONSE']._serialized_end=262
  _globals['_CLOSEREQUEST']._serialized_start=264
  _globals['_CLOSEREQUEST']._serialized_end=297
  _globals['_CLOSERESPONSE']._serialized_start=299
  _globals['_CLOSERESPONSE']._serialized_end=314
  _globals['_DETECTOR']._serialized_start=317
  _globals['_DETECTOR']._serialized_end=561
# @@protoc_insertion_point(module_scope)
