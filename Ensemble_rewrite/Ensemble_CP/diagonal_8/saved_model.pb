??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8??
?
"ann_simple_model_8/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*3
shared_name$"ann_simple_model_8/dense_16/kernel
?
6ann_simple_model_8/dense_16/kernel/Read/ReadVariableOpReadVariableOp"ann_simple_model_8/dense_16/kernel*
_output_shapes
:	?(*
dtype0
?
 ann_simple_model_8/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" ann_simple_model_8/dense_16/bias
?
4ann_simple_model_8/dense_16/bias/Read/ReadVariableOpReadVariableOp ann_simple_model_8/dense_16/bias*
_output_shapes
:(*
dtype0
?
"ann_simple_model_8/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*3
shared_name$"ann_simple_model_8/dense_17/kernel
?
6ann_simple_model_8/dense_17/kernel/Read/ReadVariableOpReadVariableOp"ann_simple_model_8/dense_17/kernel*
_output_shapes

:(*
dtype0
?
 ann_simple_model_8/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" ann_simple_model_8/dense_17/bias
?
4ann_simple_model_8/dense_17/bias/Read/ReadVariableOpReadVariableOp ann_simple_model_8/dense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
)Adam/ann_simple_model_8/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*:
shared_name+)Adam/ann_simple_model_8/dense_16/kernel/m
?
=Adam/ann_simple_model_8/dense_16/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/ann_simple_model_8/dense_16/kernel/m*
_output_shapes
:	?(*
dtype0
?
'Adam/ann_simple_model_8/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/ann_simple_model_8/dense_16/bias/m
?
;Adam/ann_simple_model_8/dense_16/bias/m/Read/ReadVariableOpReadVariableOp'Adam/ann_simple_model_8/dense_16/bias/m*
_output_shapes
:(*
dtype0
?
)Adam/ann_simple_model_8/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*:
shared_name+)Adam/ann_simple_model_8/dense_17/kernel/m
?
=Adam/ann_simple_model_8/dense_17/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/ann_simple_model_8/dense_17/kernel/m*
_output_shapes

:(*
dtype0
?
'Adam/ann_simple_model_8/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/ann_simple_model_8/dense_17/bias/m
?
;Adam/ann_simple_model_8/dense_17/bias/m/Read/ReadVariableOpReadVariableOp'Adam/ann_simple_model_8/dense_17/bias/m*
_output_shapes
:*
dtype0
?
)Adam/ann_simple_model_8/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?(*:
shared_name+)Adam/ann_simple_model_8/dense_16/kernel/v
?
=Adam/ann_simple_model_8/dense_16/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/ann_simple_model_8/dense_16/kernel/v*
_output_shapes
:	?(*
dtype0
?
'Adam/ann_simple_model_8/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/ann_simple_model_8/dense_16/bias/v
?
;Adam/ann_simple_model_8/dense_16/bias/v/Read/ReadVariableOpReadVariableOp'Adam/ann_simple_model_8/dense_16/bias/v*
_output_shapes
:(*
dtype0
?
)Adam/ann_simple_model_8/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*:
shared_name+)Adam/ann_simple_model_8/dense_17/kernel/v
?
=Adam/ann_simple_model_8/dense_17/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/ann_simple_model_8/dense_17/kernel/v*
_output_shapes

:(*
dtype0
?
'Adam/ann_simple_model_8/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/ann_simple_model_8/dense_17/bias/v
?
;Adam/ann_simple_model_8/dense_17/bias/v/Read/ReadVariableOpReadVariableOp'Adam/ann_simple_model_8/dense_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?

hidden
outlayer

dplay1

dplay2
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO

0
1
2
3

0
1
2
3
 
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
 
`^
VARIABLE_VALUE"ann_simple_model_8/dense_16/kernel(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE ann_simple_model_8/dense_16/bias&hidden/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
b`
VARIABLE_VALUE"ann_simple_model_8/dense_17/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE ann_simple_model_8/dense_17/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

=0
>1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	?total
	@count
A	variables
B	keras_api
D
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

A	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

F	variables
??
VARIABLE_VALUE)Adam/ann_simple_model_8/dense_16/kernel/mDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/ann_simple_model_8/dense_16/bias/mBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/ann_simple_model_8/dense_17/kernel/mFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE'Adam/ann_simple_model_8/dense_17/bias/mDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/ann_simple_model_8/dense_16/kernel/vDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/ann_simple_model_8/dense_16/bias/vBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/ann_simple_model_8/dense_17/kernel/vFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE'Adam/ann_simple_model_8/dense_17/bias/vDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"ann_simple_model_8/dense_16/kernel ann_simple_model_8/dense_16/bias"ann_simple_model_8/dense_17/kernel ann_simple_model_8/dense_17/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1134499
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6ann_simple_model_8/dense_16/kernel/Read/ReadVariableOp4ann_simple_model_8/dense_16/bias/Read/ReadVariableOp6ann_simple_model_8/dense_17/kernel/Read/ReadVariableOp4ann_simple_model_8/dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp=Adam/ann_simple_model_8/dense_16/kernel/m/Read/ReadVariableOp;Adam/ann_simple_model_8/dense_16/bias/m/Read/ReadVariableOp=Adam/ann_simple_model_8/dense_17/kernel/m/Read/ReadVariableOp;Adam/ann_simple_model_8/dense_17/bias/m/Read/ReadVariableOp=Adam/ann_simple_model_8/dense_16/kernel/v/Read/ReadVariableOp;Adam/ann_simple_model_8/dense_16/bias/v/Read/ReadVariableOp=Adam/ann_simple_model_8/dense_17/kernel/v/Read/ReadVariableOp;Adam/ann_simple_model_8/dense_17/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1134759
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"ann_simple_model_8/dense_16/kernel ann_simple_model_8/dense_16/bias"ann_simple_model_8/dense_17/kernel ann_simple_model_8/dense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1)Adam/ann_simple_model_8/dense_16/kernel/m'Adam/ann_simple_model_8/dense_16/bias/m)Adam/ann_simple_model_8/dense_17/kernel/m'Adam/ann_simple_model_8/dense_17/bias/m)Adam/ann_simple_model_8/dense_16/kernel/v'Adam/ann_simple_model_8/dense_16/bias/v)Adam/ann_simple_model_8/dense_17/kernel/v'Adam/ann_simple_model_8/dense_17/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1134832??
?
e
,__inference_dropout_16_layer_call_fn_1134629

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134388p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134462
input_1#
dense_16_1134450:	?(
dense_16_1134452:("
dense_17_1134456:(
dense_17_1134458:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
dropout_16/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134270?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_1134450dense_16_1134452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283?
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134294?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_17_1134456dense_17_1134458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
%__inference_signature_wrapper_1134499
input_1
unknown:	?(
	unknown_0:(
	unknown_1:(
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1134258o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134314
input_tensor#
dense_16_1134284:	?(
dense_16_1134286:("
dense_17_1134308:(
dense_17_1134310:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
dropout_16/PartitionedCallPartitionedCallinput_tensor*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134270?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_1134284dense_16_1134286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283?
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134294?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_17_1134308dense_17_1134310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?
?
*__inference_dense_16_layer_call_fn_1134588

inputs
unknown:	?(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134294

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
4__inference_ann_simple_model_8_layer_call_fn_1134512
input_tensor
unknown:	?(
	unknown_0:(
	unknown_1:(
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134314o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134270

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_1134258
input_1M
:ann_simple_model_8_dense_16_matmul_readvariableop_resource:	?(I
;ann_simple_model_8_dense_16_biasadd_readvariableop_resource:(L
:ann_simple_model_8_dense_17_matmul_readvariableop_resource:(I
;ann_simple_model_8_dense_17_biasadd_readvariableop_resource:
identity??2ann_simple_model_8/dense_16/BiasAdd/ReadVariableOp?1ann_simple_model_8/dense_16/MatMul/ReadVariableOp?2ann_simple_model_8/dense_17/BiasAdd/ReadVariableOp?1ann_simple_model_8/dense_17/MatMul/ReadVariableOpn
&ann_simple_model_8/dropout_16/IdentityIdentityinput_1*
T0*(
_output_shapes
:???????????
1ann_simple_model_8/dense_16/MatMul/ReadVariableOpReadVariableOp:ann_simple_model_8_dense_16_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype0?
"ann_simple_model_8/dense_16/MatMulMatMul/ann_simple_model_8/dropout_16/Identity:output:09ann_simple_model_8/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
2ann_simple_model_8/dense_16/BiasAdd/ReadVariableOpReadVariableOp;ann_simple_model_8_dense_16_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
#ann_simple_model_8/dense_16/BiasAddBiasAdd,ann_simple_model_8/dense_16/MatMul:product:0:ann_simple_model_8/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
#ann_simple_model_8/dense_16/SigmoidSigmoid,ann_simple_model_8/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(?
&ann_simple_model_8/dropout_17/IdentityIdentity'ann_simple_model_8/dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:?????????(?
1ann_simple_model_8/dense_17/MatMul/ReadVariableOpReadVariableOp:ann_simple_model_8_dense_17_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0?
"ann_simple_model_8/dense_17/MatMulMatMul/ann_simple_model_8/dropout_17/Identity:output:09ann_simple_model_8/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2ann_simple_model_8/dense_17/BiasAdd/ReadVariableOpReadVariableOp;ann_simple_model_8_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#ann_simple_model_8/dense_17/BiasAddBiasAdd,ann_simple_model_8/dense_17/MatMul:product:0:ann_simple_model_8/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#ann_simple_model_8/dense_17/SoftmaxSoftmax,ann_simple_model_8/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????|
IdentityIdentity-ann_simple_model_8/dense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp3^ann_simple_model_8/dense_16/BiasAdd/ReadVariableOp2^ann_simple_model_8/dense_16/MatMul/ReadVariableOp3^ann_simple_model_8/dense_17/BiasAdd/ReadVariableOp2^ann_simple_model_8/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2h
2ann_simple_model_8/dense_16/BiasAdd/ReadVariableOp2ann_simple_model_8/dense_16/BiasAdd/ReadVariableOp2f
1ann_simple_model_8/dense_16/MatMul/ReadVariableOp1ann_simple_model_8/dense_16/MatMul/ReadVariableOp2h
2ann_simple_model_8/dense_17/BiasAdd/ReadVariableOp2ann_simple_model_8/dense_17/BiasAdd/ReadVariableOp2f
1ann_simple_model_8/dense_17/MatMul/ReadVariableOp1ann_simple_model_8/dense_17/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134388

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_16_layer_call_and_return_conditional_losses_1134599

inputs1
matmul_readvariableop_resource:	?(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????(Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134545
input_tensor:
'dense_16_matmul_readvariableop_resource:	?(6
(dense_16_biasadd_readvariableop_resource:(9
'dense_17_matmul_readvariableop_resource:(6
(dense_17_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp`
dropout_16/IdentityIdentityinput_tensor*
T0*(
_output_shapes
:???????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype0?
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(h
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(g
dropout_17/IdentityIdentitydense_16/Sigmoid:y:0*
T0*'
_output_shapes
:?????????(?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0?
dense_17/MatMulMatMuldropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?

?
E__inference_dense_17_layer_call_and_return_conditional_losses_1134619

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?3
?

 __inference__traced_save_1134759
file_prefixA
=savev2_ann_simple_model_8_dense_16_kernel_read_readvariableop?
;savev2_ann_simple_model_8_dense_16_bias_read_readvariableopA
=savev2_ann_simple_model_8_dense_17_kernel_read_readvariableop?
;savev2_ann_simple_model_8_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopH
Dsavev2_adam_ann_simple_model_8_dense_16_kernel_m_read_readvariableopF
Bsavev2_adam_ann_simple_model_8_dense_16_bias_m_read_readvariableopH
Dsavev2_adam_ann_simple_model_8_dense_17_kernel_m_read_readvariableopF
Bsavev2_adam_ann_simple_model_8_dense_17_bias_m_read_readvariableopH
Dsavev2_adam_ann_simple_model_8_dense_16_kernel_v_read_readvariableopF
Bsavev2_adam_ann_simple_model_8_dense_16_bias_v_read_readvariableopH
Dsavev2_adam_ann_simple_model_8_dense_17_kernel_v_read_readvariableopF
Bsavev2_adam_ann_simple_model_8_dense_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hidden/bias/.ATTRIBUTES/VARIABLE_VALUEB*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_ann_simple_model_8_dense_16_kernel_read_readvariableop;savev2_ann_simple_model_8_dense_16_bias_read_readvariableop=savev2_ann_simple_model_8_dense_17_kernel_read_readvariableop;savev2_ann_simple_model_8_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopDsavev2_adam_ann_simple_model_8_dense_16_kernel_m_read_readvariableopBsavev2_adam_ann_simple_model_8_dense_16_bias_m_read_readvariableopDsavev2_adam_ann_simple_model_8_dense_17_kernel_m_read_readvariableopBsavev2_adam_ann_simple_model_8_dense_17_bias_m_read_readvariableopDsavev2_adam_ann_simple_model_8_dense_16_kernel_v_read_readvariableopBsavev2_adam_ann_simple_model_8_dense_16_bias_v_read_readvariableopDsavev2_adam_ann_simple_model_8_dense_17_kernel_v_read_readvariableopBsavev2_adam_ann_simple_model_8_dense_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes{
y: :	?(:(:(:: : : : : : : : : :	?(:(:(::	?(:(:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::%!

_output_shapes
:	?(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
?

?
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283

inputs1
matmul_readvariableop_resource:	?(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????(Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134634

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
f
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134673

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
H
,__inference_dropout_17_layer_call_fn_1134651

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134294`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
H
,__inference_dropout_16_layer_call_fn_1134624

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134270a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
e
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134661

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?%
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134579
input_tensor:
'dense_16_matmul_readvariableop_resource:	?(6
(dense_16_biasadd_readvariableop_resource:(9
'dense_17_matmul_readvariableop_resource:(6
(dense_17_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_16/dropout/MulMulinput_tensor!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:??????????T
dropout_16/dropout/ShapeShapeinput_tensor*
T0*
_output_shapes
:?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?(*
dtype0?
dense_16/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(h
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_17/dropout/MulMuldense_16/Sigmoid:y:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(\
dropout_17/dropout/ShapeShapedense_16/Sigmoid:y:0*
T0*
_output_shapes
:?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0?
dense_17/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134478
input_1#
dense_16_1134466:	?(
dense_16_1134468:("
dense_17_1134472:(
dense_17_1134474:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134388?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_1134466dense_16_1134468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134355?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_17_1134472dense_17_1134474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
4__inference_ann_simple_model_8_layer_call_fn_1134446
input_1
unknown:	?(
	unknown_0:(
	unknown_1:(
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?W
?
#__inference__traced_restore_1134832
file_prefixF
3assignvariableop_ann_simple_model_8_dense_16_kernel:	?(A
3assignvariableop_1_ann_simple_model_8_dense_16_bias:(G
5assignvariableop_2_ann_simple_model_8_dense_17_kernel:(A
3assignvariableop_3_ann_simple_model_8_dense_17_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: P
=assignvariableop_13_adam_ann_simple_model_8_dense_16_kernel_m:	?(I
;assignvariableop_14_adam_ann_simple_model_8_dense_16_bias_m:(O
=assignvariableop_15_adam_ann_simple_model_8_dense_17_kernel_m:(I
;assignvariableop_16_adam_ann_simple_model_8_dense_17_bias_m:P
=assignvariableop_17_adam_ann_simple_model_8_dense_16_kernel_v:	?(I
;assignvariableop_18_adam_ann_simple_model_8_dense_16_bias_v:(O
=assignvariableop_19_adam_ann_simple_model_8_dense_17_kernel_v:(I
;assignvariableop_20_adam_ann_simple_model_8_dense_17_bias_v:
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hidden/bias/.ATTRIBUTES/VARIABLE_VALUEB*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp3assignvariableop_ann_simple_model_8_dense_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp3assignvariableop_1_ann_simple_model_8_dense_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp5assignvariableop_2_ann_simple_model_8_dense_17_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp3assignvariableop_3_ann_simple_model_8_dense_17_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp=assignvariableop_13_adam_ann_simple_model_8_dense_16_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp;assignvariableop_14_adam_ann_simple_model_8_dense_16_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp=assignvariableop_15_adam_ann_simple_model_8_dense_17_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp;assignvariableop_16_adam_ann_simple_model_8_dense_17_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp=assignvariableop_17_adam_ann_simple_model_8_dense_16_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp;assignvariableop_18_adam_ann_simple_model_8_dense_16_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp=assignvariableop_19_adam_ann_simple_model_8_dense_17_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp;assignvariableop_20_adam_ann_simple_model_8_dense_17_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134422
input_tensor#
dense_16_1134410:	?(
dense_16_1134412:("
dense_17_1134416:(
dense_17_1134418:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCallinput_tensor*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134388?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_1134410dense_16_1134412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1134283?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134355?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_17_1134416dense_17_1134418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?	
f
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134355

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
4__inference_ann_simple_model_8_layer_call_fn_1134525
input_tensor
unknown:	?(
	unknown_0:(
	unknown_1:(
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:??????????
&
_user_specified_nameinput_tensor
?
?
*__inference_dense_17_layer_call_fn_1134608

inputs
unknown:(
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1134307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
4__inference_ann_simple_model_8_layer_call_fn_1134325
input_1
unknown:	?(
	unknown_0:(
	unknown_1:(
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134314o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134646

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_17_layer_call_fn_1134656

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?a
?

hidden
outlayer

dplay1

dplay2
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
P__call__
*Q&call_and_return_all_conditional_losses
R_default_save_signature"
_tf_keras_model
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
P__call__
R_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
[serving_default"
signature_map
5:3	?(2"ann_simple_model_8/dense_16/kernel
.:,(2 ann_simple_model_8/dense_16/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
4:2(2"ann_simple_model_8/dense_17/kernel
.:,2 ann_simple_model_8/dense_17/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	?total
	@count
A	variables
B	keras_api"
_tf_keras_metric
^
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
::8	?(2)Adam/ann_simple_model_8/dense_16/kernel/m
3:1(2'Adam/ann_simple_model_8/dense_16/bias/m
9:7(2)Adam/ann_simple_model_8/dense_17/kernel/m
3:12'Adam/ann_simple_model_8/dense_17/bias/m
::8	?(2)Adam/ann_simple_model_8/dense_16/kernel/v
3:1(2'Adam/ann_simple_model_8/dense_16/bias/v
9:7(2)Adam/ann_simple_model_8/dense_17/kernel/v
3:12'Adam/ann_simple_model_8/dense_17/bias/v
?2?
4__inference_ann_simple_model_8_layer_call_fn_1134325
4__inference_ann_simple_model_8_layer_call_fn_1134512
4__inference_ann_simple_model_8_layer_call_fn_1134525
4__inference_ann_simple_model_8_layer_call_fn_1134446?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkwjkwargs
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134545
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134579
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134462
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134478?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkwjkwargs
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1134258input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_16_layer_call_fn_1134588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_16_layer_call_and_return_conditional_losses_1134599?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_17_layer_call_fn_1134608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_17_layer_call_and_return_conditional_losses_1134619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dropout_16_layer_call_fn_1134624
,__inference_dropout_16_layer_call_fn_1134629?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134634
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134646?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_17_layer_call_fn_1134651
,__inference_dropout_17_layer_call_fn_1134656?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134661
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134673?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_1134499input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1134258n1?.
'?$
"?
input_1??????????
? "3?0
.
output_1"?
output_1??????????
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134462d5?2
+?(
"?
input_1??????????
p 
? "%?"
?
0?????????
? ?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134478d5?2
+?(
"?
input_1??????????
p
? "%?"
?
0?????????
? ?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134545i:?7
0?-
'?$
input_tensor??????????
p 
? "%?"
?
0?????????
? ?
O__inference_ann_simple_model_8_layer_call_and_return_conditional_losses_1134579i:?7
0?-
'?$
input_tensor??????????
p
? "%?"
?
0?????????
? ?
4__inference_ann_simple_model_8_layer_call_fn_1134325W5?2
+?(
"?
input_1??????????
p 
? "???????????
4__inference_ann_simple_model_8_layer_call_fn_1134446W5?2
+?(
"?
input_1??????????
p
? "???????????
4__inference_ann_simple_model_8_layer_call_fn_1134512\:?7
0?-
'?$
input_tensor??????????
p 
? "???????????
4__inference_ann_simple_model_8_layer_call_fn_1134525\:?7
0?-
'?$
input_tensor??????????
p
? "???????????
E__inference_dense_16_layer_call_and_return_conditional_losses_1134599]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????(
? ~
*__inference_dense_16_layer_call_fn_1134588P0?-
&?#
!?
inputs??????????
? "??????????(?
E__inference_dense_17_layer_call_and_return_conditional_losses_1134619\/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_dense_17_layer_call_fn_1134608O/?,
%?"
 ?
inputs?????????(
? "???????????
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134634^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
G__inference_dropout_16_layer_call_and_return_conditional_losses_1134646^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
,__inference_dropout_16_layer_call_fn_1134624Q4?1
*?'
!?
inputs??????????
p 
? "????????????
,__inference_dropout_16_layer_call_fn_1134629Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134661\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
G__inference_dropout_17_layer_call_and_return_conditional_losses_1134673\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? 
,__inference_dropout_17_layer_call_fn_1134651O3?0
)?&
 ?
inputs?????????(
p 
? "??????????(
,__inference_dropout_17_layer_call_fn_1134656O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
%__inference_signature_wrapper_1134499y<?9
? 
2?/
-
input_1"?
input_1??????????"3?0
.
output_1"?
output_1?????????