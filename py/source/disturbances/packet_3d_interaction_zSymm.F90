    ! this subroutine produces a single packet displaced along z. use this
    ! version, together with the z zymmetry boundary condition, to evaluate
    ! packet interactions. note that the amplitude value corresponds to a single
    ! packet, meaning that the amplitude in the spectrum will be twice this
    ! value.
    ! - giovana, 18/02/2025

    subroutine packet_3d_interaction_zSymm(nx,ny,nz,x,y,z,t,u,omega0,nmodes,beta0,bmodes,amplitude,zC)

    implicit none

    integer, intent(in) :: nx, ny, nz
    real*8, dimension(nx), intent(in) :: x
    real*8, dimension(ny), intent(in) :: y
    real*8, dimension(nz), intent(in) :: z
    real*8, intent(in) :: t
    real*8, dimension(nx,ny,nz),intent(inout) :: u
    real*8, intent(in) :: omega0, beta0
    real*8, intent(in) :: amplitude
    real*8, intent(in) :: nmodes, bmodes
    real*8, intent(in) :: zC

    integer :: ix,iz,imode,nmode,jmode,bmode
    real*8 :: pi,Tp,Tend,zf,sigmaT,sigmaX,sigmaZ,tolEnvZ,xC,spanX,spanZ
    real*8 :: envX, funcZ, funcT, envT, envZ, normMode
    complex :: iImag

    pi = 4.d0*datan(1.d0)
    Tend = (2.d0*pi/omega0)*2.5d-1

    if(t.gt.Tend) then
      return

    else
      iImag = (0.0,1.0)

      nmode=int(nmodes)
      bmode=int(bmodes)

      ! spanZ = 2.d0*z(nz)
      ! beta0 = 2.d0*pi/spanZ
      spanZ = 2.d0*pi/beta0

      xC     = (x(nx)+x(1))/2.d0
      spanX  = x(nx)-x(1)
      sigmaX = spanX/8.d0

      Tp   = Tend*5d-1
      sigmaT = Tend*2.5d-1

      tolEnvZ = 1.0d-1
      zf = spanZ/2.d0
      sigmaZ = dsqrt(-(zf**2.d0)/(2.d0*dlog(tolEnvZ)))

      ! this variable doesn't need to be recalculated every iteration, so I
      ! moved it out of the loops - giovana
      normMode = 1.d0/(nmodes*(bmodes*2.d0 + 1.d0))
      envT = dexp(-((t-Tp)**2.d0) / (2.d0*sigmaT**2.d0))

      do ix=1,nx
        envX = dexp(-(x(ix)-xC)**(2.d0)/(2.d0*sigmaX**(2.d0)))

        do iz=1,nz
          envZ = dexp(-( (z(iz)-zC)**2.d0) / (2.d0*sigmaZ**2.d0))

          do jmode=-bmode,bmode
            do imode=1,nmode

              funcT = real(exp(iImag*(imode*omega0*(t-Tp))))
              funcZ = real(exp(iImag*jmode*beta0*(z(iz)-zC)))

              u(ix,:,iz) = u(ix,:,iz) + (envT*funcT)*(envZ*funcZ)*envX

            enddo !imode=0,nmode
          enddo !jmode=-bmode,bmode
        enddo !iz=1,nz
      enddo !ix=1,nx

      u = u*amplitude*normMode

    end if
    end subroutine
