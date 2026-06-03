close all
clear all

% ============================================================
% Evaluate_Patient1_02_DebugIterations.m
% ------------------------------------------------------------
% 功能：
% 评估推理脚本保存的初始 T_view 和每一轮迭代 tumour STL。
% 默认配置为 FoundationPose 在 Patient1 / frame 02。
% 后续修改病例和帧数时，只需要改下面的配置区。
%
% 输入：
%   ../data_case/Patient{PatientNo}/{FrameNumber}/pred/debug_iterations/iter_00_tumour.stl
%   ../data_case/Patient{PatientNo}/{FrameNumber}/pred/debug_iterations/iter_XX_tumour.stl
%   ./Dataset/Patient{PatientNo}/LUS Calibration and Pose/{FrameNumber}.mat
%   ./Dataset/Patient{PatientNo}/LUS Segmentation/json/{FrameNumber}.json
%
% 输出：
%   ./Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}_debug_iterations.csv
%   ./Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}_debug_iterations.mat
%
% 直接在 DATA 目录下运行本脚本。
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
PatientTag = ['Patient', num2str(PatientNo)];
EvalTag = [MethodName, '_', PatientTag, '_', FrameNumber];
ResultRoot = fullfile(ScriptDir, 'Evaluation_Results');
ResultPath = fullfile(ResultRoot, EvalTag);
DebugStlPath = fullfile(ProjectRoot, 'data_case', PatientTag, FrameNumber, 'pred', 'debug_iterations');

if ~exist(ResultPath, 'dir')
    mkdir(ResultPath);
end

pathUSprj = fullfile(DatasetPath, PatientTag, 'LUS Calibration and Pose');
USSegPath = fullfile(DatasetPath, PatientTag, 'LUS Segmentation', 'json');
matFile = fullfile(pathUSprj, [FrameNumber, '.mat']);
jsonFile = fullfile(USSegPath, [FrameNumber, '.json']);

if ~exist(matFile, 'file')
    error(['Missing LUS calibration/pose MAT: ', matFile]);
end
if ~exist(jsonFile, 'file')
    error(['Missing LUS segmentation JSON: ', jsonFile]);
end
if ~exist(DebugStlPath, 'dir')
    error(['Missing debug STL directory: ', DebugStlPath]);
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

stlFiles = dir(fullfile(DebugStlPath, 'iter_*_tumour.stl'));
if isempty(stlFiles)
    error(['No debug tumour STL files found in: ', DebugStlPath]);
end

iters = zeros(length(stlFiles), 1);
TREs = zeros(length(stlFiles), 1);
ICs = zeros(length(stlFiles), 1);
fileNames = cell(length(stlFiles), 1);

fprintf('=== Debug iteration evaluation: Patient%d frame %s ===\n', PatientNo, FrameNumber);
for i = 1:length(stlFiles)
    name = stlFiles(i).name;
    tokens = regexp(name, 'iter_(\d+)_tumour\.stl', 'tokens');
    if isempty(tokens)
        iterNo = i - 1;
    else
        iterNo = str2double(tokens{1}{1});
    end
    stlFile = fullfile(DebugStlPath, name);
    [Points, Connectivity, n, stlName] = stlRead(stlFile);

    TumourPoints = {Points};
    TumourConnectivity = {Connectivity};
    TRE = TREcalc(P, TumourPoints, K);
    IC = ICcalc(P, TumourPoints, TumourConnectivity, OncologicMargin);

    iters(i) = iterNo;
    TREs(i) = TRE;
    ICs(i) = IC;
    fileNames{i} = name;
    fprintf('iter_%02d: TRE = %.6f mm, IC = %d, STL = %s\n', iterNo, TRE, IC, name);
end

[iters, order] = sort(iters);
TREs = TREs(order);
ICs = ICs(order);
fileNames = fileNames(order);

csvFile = fullfile(ResultPath, [EvalTag, '_debug_iterations.csv']);
fid = fopen(csvFile, 'w');
fprintf(fid, 'iter,TRE_mm,IC,file\n');
for i = 1:length(iters)
    fprintf(fid, '%d,%.10f,%d,%s\n', iters(i), TREs(i), ICs(i), fileNames{i});
end
fclose(fid);

matOut = fullfile(ResultPath, [EvalTag, '_debug_iterations.mat']);
save(matOut, 'iters', 'TREs', 'ICs', 'fileNames', 'MethodName', 'PatientNo', 'FrameNumber', 'P');

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
