%% 
clear all
close all
clc

% paths
addpath('\\nalfs002\share\MichaelChesnaye\AnalysisCode\mTRF-Toolbox\mtrf')
addpath('\\nalfs002\share\MichaelChesnaye\AnalysisCode\MatlabFunctions')

% initialise
Location        = '\\nalfs002\share\Neuromarkers_P24.01\EmmaMcGrath\';
SubjectFiles_0	= dir( Location );
SubjectFiles    = {};
for f0i=1:length(SubjectFiles_0)
    if ~contains( SubjectFiles_0(f0i).name, 'DoNotUse' ) && ...
            ( contains( SubjectFiles_0(f0i).name, 'CNH' ) || contains( SubjectFiles_0(f0i).name, 'ANH') || contains( SubjectFiles_0(f0i).name, 'CHL') )
        SubjectFiles{end+1} = SubjectFiles_0(f0i).name;
    end
end
clear SubjectFiles_0

% prepare envelope
fs100   = 100;
fs32    = 32000;
susashi	= audioread( '\\nalfs002\share\MichaelChesnaye\DATA\Stimuli\susashi_v5c.wav' );
SuInd   = 1:(fs32 * ( 0.256+0.384 ));
SaInd   = (1 + fs32 * ( 0.256+0.384 )) : ( fs32 * ( 0.256+0.384 ) * 2 );
ShiInd  = (1 + fs32 * ( 0.256+0.384 ) * 2 ) :  ( fs32 * ( 0.256+0.384 ) * 3 );
StimInd = [SuInd, ShiInd];
Sushi   = susashi(StimInd);
% envelope
env     = envelope(Sushi);              % env denotes the absolute value of the analytic signal obtained from Hilbert transform.
[b,a]   = butter(5,15/(fs32/2));
env     = filtfilt(b,a, env);
StimEnv             = downsample(env, fs32/fs100)';
StimEnv_512         = repmat( StimEnv, [1, 4] );
StimEnv_512_Norm	= ( StimEnv_512 - mean(StimEnv_512) ) / std(StimEnv_512);

% parameters and filters 
P.ARThresRange	= [50, 75, 100, 125]; 
P.Norm          = true;
fs              = 8000;
[bhigh, ahigh]  = butter(3, 1*2/fs, 'high');
[blow, alow]  	= butter(3, 15*2/fs, 'low');
fs100           = 100;

% useful
TestC = {'15.ER2','35.ER2','55.ER2','65.ER2','75.ER2','15.HA','35.HA','55.HA','65.HA','75.HA'};

% initialize
AllR        = nan( length( SubjectFiles ), length( TestC ) );
AllR2       = nan( length( SubjectFiles ), length( TestC ) );
AllLamda	= nan( length( SubjectFiles ), length( TestC ) );

% initialize cEFR
cEFR.p      = nan( length( SubjectFiles ), length( TestC ) );
cEFR.SPow   = nan( length( SubjectFiles ), length( TestC ) );
cEFR.NPow   = nan( length( SubjectFiles ), length( TestC ) );

% initialize bEFR
for phi=1:6
    bEFR.p{phi}             = nan( length( SubjectFiles ), length( TestC ) );
    bEFR.Unbiased_SPow{phi} = nan( length( SubjectFiles ), length( TestC ) );
    bEFR.Unbiased_SMag{phi}	= nan( length( SubjectFiles ), length( TestC ) );
    bEFR.NPow{phi}          = nan( length( SubjectFiles ), length( TestC ) );
end

CHECK = nan( length( SubjectFiles ), length(TestC) );
for f0i=1:length( SubjectFiles )
    
    f0i
    FileNames_mat = dir( [Location, SubjectFiles{f0i}, '\*.mat'] );
    
    for fi=1:length( FileNames_mat )
        
        % find test condition
        for ci=1:length(TestC)
            if contains( FileNames_mat(fi).name, TestC{ci} )
                This_ci = ci;
                break;
            end
        end
        clear ci
        TestCondition = TestC{This_ci};
        
        % load data
        ThisFile = [ Location, SubjectFiles{f0i}, '\', FileNames_mat(fi).name ];
        load(ThisFile)
        
        % reshape, rescale, filter
        Rec         = reshape(data', [1, length(data(:))]) / 1000;  % reshape and rescale
        Rec         = Rec * -1;
        Rec         = filtfilt(bhigh, ahigh, Rec);
        Rec         = filtfilt(blow, alow, Rec);
        P.Epochs	= reshape(Rec', [size(data,2), size(data,1)])';
        P.ELen_1pol = size(P.Epochs, 2)/2;
        P.N_PreAR   = size(P.Epochs, 1);

        % artefact rejection
        P           = ArtefactRejection_cEFR_TRF( P );  % has some small changes for TRF
        P.N_PostAR	= size( P.Epochs_Pol1, 1 );
            
        % downsample epochs
        P.Epochs_Pol1_DS = zeros( P.N_PostAR, 128);         % hard-coded
        P.Epochs_Pol2_DS = zeros( P.N_PostAR, 128);         % hard-coded
        for ei=1:P.N_PostAR
            P.Epochs_Pol1_DS(ei,:) = downsample(P.Epochs_Pol1(ei,:), fs/fs100);
            P.Epochs_Pol2_DS(ei,:) = downsample(P.Epochs_Pol2(ei,:), fs/fs100);
        end
        P.Epochs_DS = [P.Epochs_Pol1_DS, P.Epochs_Pol2_DS];
            
        % CA variants
        CA_Odd  = mean( P.Epochs_DS(1:2:end,:) );
        CA_Even = mean( P.Epochs_DS(2:2:end,:) );
        CA      = mean( P.Epochs_DS );
        CA_512  = [CA_Odd, CA_Even];
            
        %normalize by dividing by SD
        if P.Norm
            CA_512 = ( CA_512 - mean(CA_512) ) / std( CA_512 );
        end
            
        % Cross-validation
        nfold           = 5;
        [strain,rtrain]	= mTRFpartition(StimEnv_512_Norm', CA_512', nfold, 1, 'dim', 1);
        lambda          = 10.^(-6:2:0);
        cv              = mTRFcrossval(strain, rtrain, fs100, 1, -150, 450, lambda, 'zeropad', 0, 'fast', 1, 'verbose', 0);
        [rmax,idx]      = max(mean(cv.r));
            
        % Train with best lambda
        model = mTRFtrain(StimEnv_512_Norm', CA_512', fs100, 1, -150, 450, lambda(idx), 'zeropad', 0, 'verbose', 0);
            
        %test
        [~, stats_corr_mse] = mTRFpredict(StimEnv_512_Norm', CA_512', model, 'verbose', 0, 'error', 'mse');
        [~, stats_corr_mae] = mTRFpredict(StimEnv_512_Norm', CA_512', model, 'verbose', 0, 'error', 'mae');

        % store
        AllR(f0i,This_ci)        = stats_corr_mse.r;
        AllR2(f0i,This_ci)       = stats_corr_mae.r;
        AllLamda(f0i,This_ci)    = lambda(idx);

        % load cEFR results
        ThisFile_cEFR = ['\\nalfs002\share\Neuromarkers_P24.01\EmmaMcGrath\Results_cEFR\', SubjectFiles{f0i},'\Ch9_IHS_A_', strrep(TestC{This_ci}, '.', '_'), '.txt'];
        if exist( ThisFile_cEFR )
            ThisT               = readtable( ThisFile_cEFR );
            cEFR.p(f0i,This_ci)      = ThisT.HT2_p;
            cEFR.SPow(f0i,This_ci)   = ThisT.EFRPowEst_uV2_;
            cEFR.NPow(f0i,This_ci)   = ThisT.NoisePowEst_uV2_;
        end
            
        % load bEFR results
        ThisFile_bEFR = ['\\nalfs002\share\Neuromarkers_P24.01\EmmaMcGrath\Results_bEFR\', SubjectFiles{f0i},'\Ch9_IHS_A_', strrep(TestC{This_ci}, '.', '_'), '.txt'];
        if exist( ThisFile_bEFR )
            ThisT = readtable( ThisFile_bEFR );
            for phi=1:6
                bEFR.p{phi}(f0i,This_ci)             = ThisT.HT2_p(phi);
                bEFR.Unbiased_SPow{phi}(f0i,This_ci) = ThisT.FDB_UnbiasedEFRPow_nV2_(phi);
                bEFR.Unbiased_SMag{phi}(f0i,This_ci)	= ThisT.FDB_UnbiasedEFRMag_nV_(phi);
                bEFR.NPow{phi}(f0i,This_ci)          = ThisT.FDB_NoisePow_nV2_(phi);
            end
        end
        % bEFR hardcoded as s a a sh i i
            
        
    end
end

% save results
save('Results_TRF_EMMA_cEFR_bEFR', 'AllR', 'AllR2', 'AllLamda', 'cEFR', 'bEFR', 'TestC')



%% Get SII values
clear all
close all
clc

load Results_TRF_EMMA_cEFR_bEFR

% initialise
Location        = '\\nalfs002\share\Neuromarkers_P24.01\EmmaMcGrath\';
SubjectFiles_0	= dir( Location );
SubjectFiles    = {};
for f0i=1:length(SubjectFiles_0)
    if ~contains( SubjectFiles_0(f0i).name, 'DoNotUse' ) && ...
            ( contains( SubjectFiles_0(f0i).name, 'CNH' ) || contains( SubjectFiles_0(f0i).name, 'ANH') || contains( SubjectFiles_0(f0i).name, 'CHL') )
        SubjectFiles{end+1} = SubjectFiles_0(f0i).name;
    end
end
clear SubjectFiles_0

% generate csv file
% TestC = {'15.ER2','35.ER2','55.ER2','65.ER2','75.ER2','15.HA','35.HA','55.HA','65.HA','75.HA'};
% TestC = {'55.ER2','65.ER2','75.ER2','55.HA','65.HA','75.HA'};
CInd = [3:5, 8:10];         % TestC{ CInd }

% find CHL IDs
InfID = [];
for f0i=1:length(SubjectFiles)
    if contains( SubjectFiles{f0i}, 'CHL')
        InfID(end+1) = f0i;
    end
end

% load SII file
T = readtable( 'EmmaMcGrath_CHL_SII.xlsx' );

% generate headers
clear Headers NewLine
Headers{1}  = 'ID';
for ci=1:length( CInd )
    Headers{1+ci} = TestC{ CInd(ci) }
end

% generate table
fileID = fopen('SII_CHL_EmmaData.txt', 'w');
for hi=1:length(Headers)
    fprintf(fileID, '%s\t', Headers{hi});
end
fprintf(fileID, '\n');

% populate   
for f0i=1:length( InfID )
	NewLine{1} = SubjectFiles{ InfID(f0i) };
    
    % find table entry
    for si=1:length( T.Participant )
        if contains( T.Participant{si}, SubjectFiles{ InfID(f0i) } ) 
            break
        end
    end
    
    % 
    NewLine{2} = num2str( T.Unaided_55(si) );
    NewLine{3} = num2str( T.Unaided_65(si) );
    NewLine{4} = num2str( T.Unaided_75(si) );
    NewLine{5} = num2str( T.Aided_55(si) );
    NewLine{6} = num2str( T.Aided_65(si) );
    NewLine{7} = num2str( T.Aided_75(si) );
    
    WriteLine(NewLine, fileID)
end
pause(0.1); fclose(fileID); pause(0.1)

%% Reorganize


%% generate .txt file
% ...
% ...
% ...
% ...
% ...
% ...

%% TRF predictions
clear all
close all
clc

load Results_TRF_EMMA_cEFR_bEFR

% SII values 
TestC = {'15.ER2','35.ER2','55.ER2','65.ER2','75.ER2','15.HA','35.HA','55.HA','65.HA','75.HA'};

% ...
% ...
% ...
% ...
% ...
% ...



