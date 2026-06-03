close all
clear all

% ============================================================
% Evaluate_Patient1_02_CoordinateDebug.m
% ------------------------------------------------------------
% 独立 debug 脚本，不参与正式流程。
%
% 目的：
% 检查 Evaluation.m / MATLAB / STL 阶段是否还需要额外坐标系转换。
% 对指定方法的输出 STL 测试多种点坐标变换。
% 默认配置为 FoundationPose 在 Patient1 / frame 02。
% 后续修改病例和帧数时，只需要改下面的配置区。
%   identity
%   flip_x / flip_y / flip_z
%   flip_xy / flip_xz / flip_yz / flip_xyz
%   original_eval_yz_flip，即原始 Evaluation.m 注释中的 Points(:,2:3)=-Points(:,2:3)
%
% 输出：
%   DATA/Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}_coordinate_debug.csv
%   DATA/Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}_coordinate_debug.mat
%
% 判断：
% - 如果 identity 明显最好，评估阶段不需要额外转换。
% - 如果 original_eval_yz_flip 明显最好，说明原始 Evaluation.m 的 OpenGL->Matlab 转换可能需要打开。
% - 如果某个 det=-1 单轴/三轴 flip 最好，说明可能存在镜像/左右手问题。
% ============================================================

ScriptDir = fileparts(mfilename('fullpath'));
ProjectRoot = fileparts(ScriptDir);
addpath(fullfile(ScriptDir, 'stlTools'));

% ---------------- User configuration ----------------
MethodName = 'FoundationPose';
PatientNo = 1;
FrameNumber = '02';
K = 10;
M = 50;
OncologicMargin = 10;

DatasetPath = fullfile(ScriptDir, 'Dataset');
RegistrationRoot = fullfile(ScriptDir, 'Registration Methods');
PatientTag = ['Patient', num2str(PatientNo)];
EvalTag = [MethodName, '_', PatientTag, '_', FrameNumber];
ResultRoot = fullfile(ScriptDir, 'Evaluation_Results');
ResultPath = fullfile(ResultRoot, EvalTag);

if ~exist(ResultPath, 'dir')
    mkdir(ResultPath);
end

pathUSprj = fullfile(DatasetPath, PatientTag, 'LUS Calibration and Pose');
regPath = fullfile(RegistrationRoot, MethodName, PatientTag);
USSegPath = fullfile(DatasetPath, PatientTag, 'LUS Segmentation', 'json');

stlFile = fullfile(regPath, [FrameNumber, '.stl']);
matFile = fullfile(pathUSprj, [FrameNumber, '.mat']);
jsonFile = fullfile(USSegPath, [FrameNumber, '.json']);

generatedStlFile = fullfile(ProjectRoot, 'result', PatientTag, FrameNumber, ...
    'evaluation', 'Registration Methods', MethodName, PatientTag, [FrameNumber, '.stl']);
if ~exist(stlFile, 'file') && exist(generatedStlFile, 'file')
    if ~exist(regPath, 'dir')
        mkdir(regPath);
    end
    copyfile(generatedStlFile, stlFile);
    fprintf('[Info] Copied generated STL to evaluation input: %s\n', stlFile);
end

if ~exist(stlFile, 'file')
    error(['Missing registration STL: ', stlFile]);
end
if ~exist(matFile, 'file')
    error(['Missing LUS calibration/pose MAT: ', matFile]);
end
if ~exist(jsonFile, 'file')
    error(['Missing LUS segmentation JSON: ', jsonFile]);
end

fid = fopen(jsonFile);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);
USprofile = getTumourProfileFromJson(data);
load(matFile);

USprofile = [SLus * USprofile(1,:); zeros(size(USprofile(1,:))); SAus * USprofile(2,:)];
USprofile_probe = (Rus * USprofile) + Tus;
USprofile_cam = (Rpr * USprofile_probe + Tpr);
P = zeros(3, M, 1);
P(:,:,1) = SampleProfile(USprofile_cam, M);

[PointsRaw, Connectivity, n, name] = stlRead(stlFile);

Transforms = struct();
Transforms.identity = [1 1 1];
Transforms.flip_x = [-1 1 1];
Transforms.flip_y = [1 -1 1];
Transforms.flip_z = [1 1 -1];
Transforms.flip_xy = [-1 -1 1];
Transforms.flip_xz = [-1 1 -1];
Transforms.flip_yz = [1 -1 -1];
Transforms.flip_xyz = [-1 -1 -1];
Transforms.original_eval_yz_flip = [1 -1 -1];

names = fieldnames(Transforms);
TREs = zeros(length(names), 1);
ICs = zeros(length(names), 1);
dets = zeros(length(names), 1);

fprintf('=== Coordinate debug evaluation: Patient%d frame %s ===\n', PatientNo, FrameNumber);
fprintf('STL: %s\n\n', stlFile);

for i = 1:length(names)
    transformName = names{i};
    signs = Transforms.(transformName);
    Points = PointsRaw;
    Points(:,1) = signs(1) * Points(:,1);
    Points(:,2) = signs(2) * Points(:,2);
    Points(:,3) = signs(3) * Points(:,3);

    TumourPoints = {Points};
    TumourConnectivity = {Connectivity};
    TRE = TREcalc(P, TumourPoints, K);
    IC = ICcalc(P, TumourPoints, TumourConnectivity, OncologicMargin);
    detVal = prod(signs);

    TREs(i) = TRE;
    ICs(i) = IC;
    dets(i) = detVal;
    fprintf('%-22s det=%+d TRE=%.6f mm IC=%d\n', transformName, detVal, TRE, IC);
end

[bestTRE, bestIdx] = min(TREs);
fprintf('\nBest by TRE: %s, TRE=%.6f mm, IC=%d, det=%+d\n', names{bestIdx}, bestTRE, ICs(bestIdx), dets(bestIdx));

csvFile = fullfile(ResultPath, [EvalTag, '_coordinate_debug.csv']);
fid = fopen(csvFile, 'w');
fprintf(fid, 'transform,det,TRE_mm,IC\n');
for i = 1:length(names)
    fprintf(fid, '%s,%d,%.10f,%d\n', names{i}, dets(i), TREs(i), ICs(i));
end
fclose(fid);

matOut = fullfile(ResultPath, [EvalTag, '_coordinate_debug.mat']);
save(matOut, 'names', 'dets', 'TREs', 'ICs', 'bestIdx', 'MethodName', 'PatientNo', 'FrameNumber', 'stlFile');

fprintf('\n[OK] CSV: %s\n', csvFile);
fprintf('[OK] MAT: %s\n', matOut);


function USprofile = getTumourProfileFromJson(data)
    if isfield(data, 'objects')
        objects = data.objects;
    else
        error('JSON does not contain objects field.');
    end

    tumourObj = [];
    if isstruct(objects)
        for idx = 1:numel(objects)
            obj = objects(idx);
            if isfield(obj, 'classTitle') && strcmpi(obj.classTitle, 'Tumor')
                tumourObj = obj;
                break;
            end
        end
        if isempty(tumourObj)
            tumourObj = objects(1);
        end
    else
        error('Unsupported JSON objects format.');
    end

    exterior = tumourObj.points.exterior;
    if iscell(exterior)
        pts = cell2mat(exterior);
    else
        pts = exterior;
    end
    USprofile = pts';
end


function IC = ICcalc(P, TumourPoints, TumourConnectivity, OncologicMargin)
N = length(TumourPoints);
IC = 1;
for i=1:N
    T = TumourPoints{i};
    Tc = mean(T);
    TumorPointsAug = zeros(size(T));
    for j=1:length(T)
        Point = T(j,:);
        TumorPointsAug(j,:) = Point + OncologicMargin * (Point-Tc)/vec_norm(Point-Tc);
    end
    IC = IC & all(intriangulation(TumorPointsAug,TumourConnectivity{i},P(:,:,i)'));
end
end


function TRE = TREcalc(P, TumourPoints, K)
N = length(TumourPoints);
R0=eye(3);
t0=[];
for i=1:N
    CoG_P = mean(P(:,:,i),2);
    CoG_T = mean(TumourPoints{i}',2);
    t0 = [t0,CoG_T-CoG_P];
end
t0 = mean(t0,2);

R = R0;
t = t0;
for k=1:K
    T = ClosetPoint(TumourPoints, P, R, t);
    [regParams,Pfit,ErrorStats] = absor(P(:,:),T(:,:));
    R = regParams.R;
    t = regParams.t;
end

P = P(:,:);
T = T(:,:);
TRE = mean(vec_norm(P-T));
end


function PSampled = SampleProfile(P,M)
Perimeter = sum(vec_norm(P(:,2:end) - P(:,1:end-1)));
delta_P = Perimeter / M;
PSampled(:,1) = P(:,1);
Next = 2;
CurrentPoint = P(:,1);
NextPoint = P(:,2);
Distance2NextPoint = norm(NextPoint-CurrentPoint);
for i = 2:M
    delta_Pi = delta_P;
    while delta_Pi>Distance2NextPoint
        Next = Next + 1;
        delta_Pi = delta_Pi - Distance2NextPoint;
        CurrentPoint = P(:,Next-1);
        NextPoint = P(:,Next);
        Distance2NextPoint = norm(NextPoint-CurrentPoint);
    end
    CurrentPoint = CurrentPoint + delta_Pi * (NextPoint-CurrentPoint)./norm(NextPoint-CurrentPoint);
    Distance2NextPoint = norm(NextPoint-CurrentPoint);
    PSampled(:,i) = CurrentPoint;
end
end


function TCP = ClosetPoint(TumourPoints, P, R, t)
    M = size(P,2);
    N = size(P,3);
    TCP = zeros(3,M,N);
    for i=1:N
        for j=1:M
            Idx = knnsearch(TumourPoints{i}, (R*P(:,j,i)+t)');
            TCP(:,j,i)=(TumourPoints{i}(Idx,:))';
        end
    end
end


function normX = vec_norm(X)
    normX = sqrt(sum(X.^2));
end
