# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: platform.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='platform.proto',
  package='protos',
  syntax='proto3',
  serialized_pb=_b('\n\x0eplatform.proto\x12\x06protos\",\n\nFitRequest\x12\r\n\x05title\x18\x01 \x03(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x03(\t\"\x1f\n\rFilterRequest\x12\x0e\n\x06tokens\x18\x01 \x03(\t\" \n\x0cQueryRequest\x12\x10\n\x08keywords\x18\x01 \x03(\t\"/\n\x0e\x43ommonResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\".\n\x0e\x46ilterResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0e\n\x06tokens\x18\x02 \x03(\t\"F\n\rQueryResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x10\n\x08keywords\x18\x02 \x03(\t\x12\x15\n\rprobabilities\x18\x03 \x03(\x02\x32\xb4\x01\n\x08Platform\x12\x35\n\x03\x46it\x12\x12.protos.FitRequest\x1a\x16.protos.CommonResponse\"\x00(\x01\x12\x39\n\x06\x46ilter\x12\x15.protos.FilterRequest\x1a\x16.protos.FilterResponse\"\x00\x12\x36\n\x05Query\x12\x14.protos.QueryRequest\x1a\x15.protos.QueryResponse\"\x00\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_FITREQUEST = _descriptor.Descriptor(
  name='FitRequest',
  full_name='protos.FitRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='title', full_name='protos.FitRequest.title', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='content', full_name='protos.FitRequest.content', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26,
  serialized_end=70,
)


_FILTERREQUEST = _descriptor.Descriptor(
  name='FilterRequest',
  full_name='protos.FilterRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tokens', full_name='protos.FilterRequest.tokens', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=103,
)


_QUERYREQUEST = _descriptor.Descriptor(
  name='QueryRequest',
  full_name='protos.QueryRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='keywords', full_name='protos.QueryRequest.keywords', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=105,
  serialized_end=137,
)


_COMMONRESPONSE = _descriptor.Descriptor(
  name='CommonResponse',
  full_name='protos.CommonResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='protos.CommonResponse.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='message', full_name='protos.CommonResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=139,
  serialized_end=186,
)


_FILTERRESPONSE = _descriptor.Descriptor(
  name='FilterResponse',
  full_name='protos.FilterResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='protos.FilterResponse.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tokens', full_name='protos.FilterResponse.tokens', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=188,
  serialized_end=234,
)


_QUERYRESPONSE = _descriptor.Descriptor(
  name='QueryResponse',
  full_name='protos.QueryResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='protos.QueryResponse.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='keywords', full_name='protos.QueryResponse.keywords', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='probabilities', full_name='protos.QueryResponse.probabilities', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=236,
  serialized_end=306,
)

DESCRIPTOR.message_types_by_name['FitRequest'] = _FITREQUEST
DESCRIPTOR.message_types_by_name['FilterRequest'] = _FILTERREQUEST
DESCRIPTOR.message_types_by_name['QueryRequest'] = _QUERYREQUEST
DESCRIPTOR.message_types_by_name['CommonResponse'] = _COMMONRESPONSE
DESCRIPTOR.message_types_by_name['FilterResponse'] = _FILTERRESPONSE
DESCRIPTOR.message_types_by_name['QueryResponse'] = _QUERYRESPONSE

FitRequest = _reflection.GeneratedProtocolMessageType('FitRequest', (_message.Message,), dict(
  DESCRIPTOR = _FITREQUEST,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.FitRequest)
  ))
_sym_db.RegisterMessage(FitRequest)

FilterRequest = _reflection.GeneratedProtocolMessageType('FilterRequest', (_message.Message,), dict(
  DESCRIPTOR = _FILTERREQUEST,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.FilterRequest)
  ))
_sym_db.RegisterMessage(FilterRequest)

QueryRequest = _reflection.GeneratedProtocolMessageType('QueryRequest', (_message.Message,), dict(
  DESCRIPTOR = _QUERYREQUEST,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.QueryRequest)
  ))
_sym_db.RegisterMessage(QueryRequest)

CommonResponse = _reflection.GeneratedProtocolMessageType('CommonResponse', (_message.Message,), dict(
  DESCRIPTOR = _COMMONRESPONSE,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.CommonResponse)
  ))
_sym_db.RegisterMessage(CommonResponse)

FilterResponse = _reflection.GeneratedProtocolMessageType('FilterResponse', (_message.Message,), dict(
  DESCRIPTOR = _FILTERRESPONSE,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.FilterResponse)
  ))
_sym_db.RegisterMessage(FilterResponse)

QueryResponse = _reflection.GeneratedProtocolMessageType('QueryResponse', (_message.Message,), dict(
  DESCRIPTOR = _QUERYRESPONSE,
  __module__ = 'platform_pb2'
  # @@protoc_insertion_point(class_scope:protos.QueryResponse)
  ))
_sym_db.RegisterMessage(QueryResponse)


import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities


class PlatformStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Fit = channel.stream_unary(
        '/protos.Platform/Fit',
        request_serializer=FitRequest.SerializeToString,
        response_deserializer=CommonResponse.FromString,
        )
    self.Filter = channel.unary_unary(
        '/protos.Platform/Filter',
        request_serializer=FilterRequest.SerializeToString,
        response_deserializer=FilterResponse.FromString,
        )
    self.Query = channel.unary_unary(
        '/protos.Platform/Query',
        request_serializer=QueryRequest.SerializeToString,
        response_deserializer=QueryResponse.FromString,
        )


class PlatformServicer(object):

  def Fit(self, request_iterator, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Filter(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Query(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_PlatformServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Fit': grpc.stream_unary_rpc_method_handler(
          servicer.Fit,
          request_deserializer=FitRequest.FromString,
          response_serializer=CommonResponse.SerializeToString,
      ),
      'Filter': grpc.unary_unary_rpc_method_handler(
          servicer.Filter,
          request_deserializer=FilterRequest.FromString,
          response_serializer=FilterResponse.SerializeToString,
      ),
      'Query': grpc.unary_unary_rpc_method_handler(
          servicer.Query,
          request_deserializer=QueryRequest.FromString,
          response_serializer=QueryResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'protos.Platform', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class BetaPlatformServicer(object):
  """The Beta API is deprecated for 0.15.0 and later.

  It is recommended to use the GA API (classes and functions in this
  file not marked beta) for all further purposes. This class was generated
  only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
  def Fit(self, request_iterator, context):
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
  def Filter(self, request, context):
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
  def Query(self, request, context):
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


class BetaPlatformStub(object):
  """The Beta API is deprecated for 0.15.0 and later.

  It is recommended to use the GA API (classes and functions in this
  file not marked beta) for all further purposes. This class was generated
  only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
  def Fit(self, request_iterator, timeout, metadata=None, with_call=False, protocol_options=None):
    raise NotImplementedError()
  Fit.future = None
  def Filter(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    raise NotImplementedError()
  Filter.future = None
  def Query(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    raise NotImplementedError()
  Query.future = None


def beta_create_Platform_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
  """The Beta API is deprecated for 0.15.0 and later.

  It is recommended to use the GA API (classes and functions in this
  file not marked beta) for all further purposes. This function was
  generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
  request_deserializers = {
    ('protos.Platform', 'Filter'): FilterRequest.FromString,
    ('protos.Platform', 'Fit'): FitRequest.FromString,
    ('protos.Platform', 'Query'): QueryRequest.FromString,
  }
  response_serializers = {
    ('protos.Platform', 'Filter'): FilterResponse.SerializeToString,
    ('protos.Platform', 'Fit'): CommonResponse.SerializeToString,
    ('protos.Platform', 'Query'): QueryResponse.SerializeToString,
  }
  method_implementations = {
    ('protos.Platform', 'Filter'): face_utilities.unary_unary_inline(servicer.Filter),
    ('protos.Platform', 'Fit'): face_utilities.stream_unary_inline(servicer.Fit),
    ('protos.Platform', 'Query'): face_utilities.unary_unary_inline(servicer.Query),
  }
  server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
  return beta_implementations.server(method_implementations, options=server_options)


def beta_create_Platform_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
  """The Beta API is deprecated for 0.15.0 and later.

  It is recommended to use the GA API (classes and functions in this
  file not marked beta) for all further purposes. This function was
  generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
  request_serializers = {
    ('protos.Platform', 'Filter'): FilterRequest.SerializeToString,
    ('protos.Platform', 'Fit'): FitRequest.SerializeToString,
    ('protos.Platform', 'Query'): QueryRequest.SerializeToString,
  }
  response_deserializers = {
    ('protos.Platform', 'Filter'): FilterResponse.FromString,
    ('protos.Platform', 'Fit'): CommonResponse.FromString,
    ('protos.Platform', 'Query'): QueryResponse.FromString,
  }
  cardinalities = {
    'Filter': cardinality.Cardinality.UNARY_UNARY,
    'Fit': cardinality.Cardinality.STREAM_UNARY,
    'Query': cardinality.Cardinality.UNARY_UNARY,
  }
  stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
  return beta_implementations.dynamic_stub(channel, 'protos.Platform', cardinalities, options=stub_options)
# @@protoc_insertion_point(module_scope)
