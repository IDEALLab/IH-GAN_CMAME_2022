function [xPhys, yPhys, zPhys, output] = top3d_sdf_E_rho_nu_cmp(nelx,nely,nelz,volfrac,rmin)
% USER-DEFINED LOOP PARAMETERS
maxloop = inf;    % Maximum number of iterations
tolx = 0.00001;      % Terminarion criterion
displayflag = 0;  % Display structure flag
% USER-DEFINED MATERIAL PROPERTIES
E0 = 41.7;         % Initial Young's modulus
nu0 = 0.2778;       % Initial Poisson's ratio
rho0 = 0.5014;      % Initial density
% E0 = 30.17;         % Initial Young's modulus
% nu0 = 0.3075;       % Initial Poisson's ratio
% rho0 = 0.4513;      % Initial density
% USER-DEFINED LOAD DOFs
[il,jl,kl] = meshgrid(nelx, 0, 0:nelz);                 % Coordinates
loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl); % Node IDs
loaddof = 3*loadnid(:) - 1;                             % DOFs
% USER-DEFINED SUPPORT FIXED DOFs
[iif,jf,kf] = meshgrid(0,0:nely,0:nelz);                  % Coordinates
fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf); % Node IDs
fixeddof = [3*fixednid(:); 3*fixednid(:)-1; 3*fixednid(:)-2]; % DOFs
% PREPARE FINITE ELEMENT ANALYSIS
nele = nelx*nely*nelz; % Number of elements
ndof = 3*(nelx+1)*(nely+1)*(nelz+1); % Total number of degree of freedom
F = sparse(loaddof,1,-100,ndof,1); % Apply loading condition
U = zeros(ndof,1); % Initialize displacement
freedofs = setdiff(1:ndof,fixeddof); % Indices indicating unconstrained DOFs (excluding the fixed DOFs)
% KE = lk_H8(nu); % Element stiffness matrix (24 by 24)
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);
nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeids = repmat(nodeids,size(nodeidz))+repmat(nodeidz,size(nodeids));
edofVec = 3*nodeids(:)+1;
edofMat = repmat(edofVec,1,24)+ ...
    repmat([0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1 ...
    3*(nely+1)*(nelx+1)+[0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1]],nele,1); % Node indices for each element following the local node order (nele by 24)
iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1); % Indices for global stiffness matrix
% PREPARE FILTER
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1); % DO NOT UNDERSTAND!!!
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1;
            for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2;
                        k = k+1;
                        iH(k) = e1; % Element i
                        jH(k) = e2; % Surrounding element j
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2)); % Filter kernel/weight factor
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH); % Weight factor for each element with surrounding elements (nele by nele)
Hs = sum(H,2); % Denominator of Equation 21
% INITIALIZE ITERATION
X = zeros(nely, 3*nelx, nelz);
x = repmat(E0,[nely,nelx,nelz]);   % Initialize Young's modulus at each site
y = repmat(nu0,[nely,nelx,nelz]); % Initialize Poisson's ratio
z = repmat(rho0,[nely,nelx,nelz]);  % Initialize volume fraction at each site
X(:,1:nelx,:)=x;
X(:,nelx+1:2*nelx,:)=y;
X(:,2*nelx+1:3*nelx,:)=z;
xPhys = x;
yPhys = y;
zPhys = z;
% global ce % Shared between myfun and myHessianFcn
A = [];
B = [];
Aeq = [];
Beq = [];
% LB = [20*ones(size(x)), 0.23*ones(size(y)), 0.26*ones(size(z))]; % Lower bound of the design variable
LB = [0.01*ones(size(x)), 0.01*ones(size(y)), 0.01*ones(size(z))]; % Lower bound of the design variable
UB = [128*ones(size(x)),  0.328*ones(size(y)), ones(size(z))]; % Upper bound of the design variable
% options = optimset('Algorithm','interior-point','TolX',tolx,'MaxIter',maxloop, 'MaxFunEvals',inf, 'Display','none', 'OutputFcn',@(X,optimValues,state) myOutputFcn(X,optimValues,state,displayflag), 'PlotFcns',@optimplotfval);
options = optimoptions(@fmincon,'Display','iter','Algorithm','interior-point','StepTolerance',tolx,'MaxIterations',maxloop,'MaxFunctionEvaluations',inf,'ConstraintTolerance',1e-10,'HessianApproximation','bfgs',...
                       'OutputFcn',@(X,optimValues,state) myOutputFcn(X,optimValues,state,displayflag),'PlotFcn',@optimplotfval);
% OPTIONS = optimset('TolX',tolx, 'MaxIter',maxloop, 'Algorithm','interior-point',...
% 'GradObj','on', 'GradConstr','on', 'Hessian','off', 'HessFcn','bfgs',...
% 'Display','none', 'OutputFcn',@(X,optimValues,state) myOutputFcn(X,optimValues,state,displayflag), 'PlotFcns',@optimplotfval);

function f = myObjFcn(X)
    x = X(:,1:nelx,:);
    y = X(:,nelx+1:2*nelx,:);
    z = X(:,2*nelx+1:3*nelx,:);
    xPhys(:) = (H*x(:))./Hs;
    yPhys(:) = (H*y(:))./Hs;
    zPhys(:) = (H*z(:))./Hs;
    xPhysf = xPhys(:);
    yPhysf = yPhys(:);
    zPhysf = zPhys(:); 
    sK = [];
    for i = 1:nely*nelx*nelz
        KE = lk_H8(yPhysf(i));
        sK = horzcat(sK, KE(:)*xPhysf(i));
    end
    % FE-ANALYSIS
%     sK = reshape(KE*repmat((Emin+xPhys(:)'.^penal*(E0-Emin)),[nele,1]),24*24*nele,1);
    sK = reshape(sK,24*24*nele,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    UE = U(edofMat);
    ce = [];
    for j = 1:nely*nelx*nelz
        ce = [ce; (UE(j,:)*lk_H8(yPhysf(i))).*UE(j,:)];
    end
    ce = reshape(sum(ce,2),[nely,nelx,nelz]);
%     ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),[nely,nelx,nelz]);
    c = sum(sum(sum(xPhys.*ce)));
%     dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
%     % FILTERING AND MODIFICATION OF SENSITIVITIES
%     dc(:) = H*(dc(:)./Hs);
    % RETURN
    f = c + 0*(sum(zPhysf) - volfrac*nele)^2;
%     gradf = dc(:);
end % myfun

function [cneq, ceq, gradc, gradceq] = myConstrFcn(X)
    x = X(:,1:nelx,:);
    y = X(:,nelx+1:2*nelx,:);
    z = X(:,2*nelx+1:3*nelx,:);
    xPhys(:) = (H*x(:))./Hs;
    yPhys(:) = (H*y(:))./Hs;
    zPhys(:) = (H*z(:))./Hs;   
    % Non-linear Constraints
    % Load material properties
    mat_prp = load('.\data\mat_prp.mat');
    mat_prp = mat_prp.mat_prp;
    % Load average point spacing
    rr_ave = load('.\data\rr_ave_3d.mat');
    rr_ave = rr_ave.rr_ave;
    % Load average position of neighboring points
    p_bar = load('.\data\p_bar_3d.mat');
    p_bar = p_bar.ans;
    % Find maximum E for normalization 
    xs = mat_prp(:,1); xs_max = max(xs);
    % Search for the nearest neighbor
    p_ne = zeros(nelx*nely*nelz,3);
    for i = 1:nelx*nely*nelz
        kid = dsearchn(p_bar, [xPhys(i)/xs_max, yPhys(i), zPhys(i)]);
        p_ne(i,:) = p_bar(kid, :);
    end
    % Nearest neighboring points
%     p_ne_ave(isnan(p_ne_ave))=0;
    xp = p_ne(:,1);yp = p_ne(:,2);zp = p_ne(:,3);
    % Implicit signed distance field for nonlinear constraints
    cneq = [];
    for i = 1:nelx*nely*nelz
        cneq1 = 1000*(((xPhys(i)/xs_max-xp(i))^2 + (yPhys(i)-yp(i))^2 + (zPhys(i)-zp(i))^2)^(0.5) - 2.0*rr_ave);
        cneq = [cneq cneq1];
    end
    cneq2 = sum(zPhys(:)) - volfrac*nele;
    cneq = [cneq cneq2];
    gradc = [];
    % Linear Constraints
    ceq     = [];
    gradceq = [];
end % mycon

function stop = myOutputFcn(X,optimValues,state,displayflag)
    stop = false;
    switch state
        case 'iter'
            % Make updates to plot or guis as needed
            x = X(:,1:nelx,:);
            y = X(:,nelx+1:2*nelx,:);
            z = X(:,2*nelx+1:3*nelx,:);
            xPhys = reshape(x, nely, nelx, nelz);
            yPhys = reshape(y, nely, nelx, nelz);
            zPhys = reshape(z, nely, nelx, nelz);
            %% PRINT RESULTS
            fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',optimValues.iteration,optimValues.fval, ...
                mean(zPhys(:)),optimValues.stepsize);
            %% PLOT DENSITIES
            if displayflag, figure(10); clf; display_3D(zPhys); end
            title([' It.:',sprintf('%5i',optimValues.iteration),...
                ' Obj. = ',sprintf('%11.4f',optimValues.fval),...
                ' ch.:',sprintf('%7.3f',optimValues.stepsize)]);
        case 'init'
            % Setup for plots or guis
            if displayflag
                figure(10)
            end
        case 'done'
            % Cleanup of plots, guis, or final plot
            figure(10); clf; display_3D(zPhys);
        otherwise
    end % switch
end % myOutputFcn

output = fmincon(@myObjFcn, X, A, B, Aeq, Beq, LB, UB, @myConstrFcn, options);
end

% === GENERATE ELEMENT STIFFNESS MATRIX ===
function [KE] = lk_H8(nu)
A = [32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8;
    -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12];
k = 1/144*A'*[1; nu];

K1 = [k(1) k(2) k(2) k(3) k(5) k(5);
    k(2) k(1) k(2) k(4) k(6) k(7);
    k(2) k(2) k(1) k(4) k(7) k(6);
    k(3) k(4) k(4) k(1) k(8) k(8);
    k(5) k(6) k(7) k(8) k(1) k(2);
    k(5) k(7) k(6) k(8) k(2) k(1)];
K2 = [k(9)  k(8)  k(12) k(6)  k(4)  k(7);
    k(8)  k(9)  k(12) k(5)  k(3)  k(5);
    k(10) k(10) k(13) k(7)  k(4)  k(6);
    k(6)  k(5)  k(11) k(9)  k(2)  k(10);
    k(4)  k(3)  k(5)  k(2)  k(9)  k(12)
    k(11) k(4)  k(6)  k(12) k(10) k(13)];
K3 = [k(6)  k(7)  k(4)  k(9)  k(12) k(8);
    k(7)  k(6)  k(4)  k(10) k(13) k(10);
    k(5)  k(5)  k(3)  k(8)  k(12) k(9);
    k(9)  k(10) k(2)  k(6)  k(11) k(5);
    k(12) k(13) k(10) k(11) k(6)  k(4);
    k(2)  k(12) k(9)  k(4)  k(5)  k(3)];
K4 = [k(14) k(11) k(11) k(13) k(10) k(10);
    k(11) k(14) k(11) k(12) k(9)  k(8);
    k(11) k(11) k(14) k(12) k(8)  k(9);
    k(13) k(12) k(12) k(14) k(7)  k(7);
    k(10) k(9)  k(8)  k(7)  k(14) k(11);
    k(10) k(8)  k(9)  k(7)  k(11) k(14)];
K5 = [k(1) k(2)  k(8)  k(3) k(5)  k(4);
    k(2) k(1)  k(8)  k(4) k(6)  k(11);
    k(8) k(8)  k(1)  k(5) k(11) k(6);
    k(3) k(4)  k(5)  k(1) k(8)  k(2);
    k(5) k(6)  k(11) k(8) k(1)  k(8);
    k(4) k(11) k(6)  k(2) k(8)  k(1)];
K6 = [k(14) k(11) k(7)  k(13) k(10) k(12);
    k(11) k(14) k(7)  k(12) k(9)  k(2);
    k(7)  k(7)  k(14) k(10) k(2)  k(9);
    k(13) k(12) k(10) k(14) k(7)  k(11);
    k(10) k(9)  k(2)  k(7)  k(14) k(7);
    k(12) k(2)  k(9)  k(11) k(7)  k(14)];
KE = 1/((nu+1)*(1-2*nu))*...
    [ K1  K2  K3  K4;
    K2'  K5  K6  K3';
    K3' K6  K5' K2';
    K4  K3  K2  K1'];
end
% === DISPLAY 3D TOPOLOGY (ISO-VIEW) ===
function display_3D(rho)
[nely,nelx,nelz] = size(rho);
hx = 1; hy = 1; hz = 1;            % User-defined unit element size
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'color','w','Name','ISO display','NumberTitle','off');
for k = 1:nelz
    z = (k-1)*hz;
    for i = 1:nelx
        x = (i-1)*hx;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
            if (rho(j,i,k) > 0.10)  % User-defined display density threshold
                vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'LineStyle','none','FaceColor',[0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k))],'FaceAlpha',.2);
                hold on;
            end
        end
    end
end
axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6);
end