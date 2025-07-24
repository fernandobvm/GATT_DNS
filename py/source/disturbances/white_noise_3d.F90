    subroutine white_noise_3d(nx,ny,nz,x,y,z,t,u,omega0,nmodes,beta0,bmodes,amplitude)

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

    integer :: ix,iz,imode,nmode,jmode,bmode
    real*8 :: pi,Tp,zf,sigmaT,sigmaX,sigmaZ,tolEnvZ,xC,spanX,spanZ
    real*8 :: envX, envT, envZ, normMode
    real*8, dimension(int(nmodes),int(bmodes*2+1)) :: phi
    integer :: status, i, j
    complex :: iImag, funcZ, funcT, phase

    ! Read phases from phi.dat file
    OPEN(UNIT=32,FILE="phi.dat",ACTION="READ",IOSTAT=status)
    IF(status == 0) THEN

      DO i = 1,(int(nmodes))
        READ(32,*,IOSTAT=status) (phi(i,j),j=1,(int(bmodes)*2+1))

        IF(status/=0) THEN
          WRITE(*,120) i
          120 FORMAT("Error found in line ",I4," of phi.dat file.")
          STOP
        END IF

      END DO

    ELSE
      WRITE(*,110)
      110 FORMAT("Error: phi.dat not found.")
      STOP

    END IF

    CLOSE(UNIT=32)

    ! now, actually calculate the white noise
    iImag = (0.0,1.0)
    pi = 4.d0*datan(1.d0)

    nmode=int(nmodes)
    bmode=int(bmodes)

    ! spanZ = 2.d0*z(nz)
    ! beta0 = 2.d0*pi/spanZ
    spanZ = 2.d0*pi/beta0

    xC     = (x(nx)+x(1))/2.d0
    spanX  = x(nx)-x(1)
    sigmaX = spanX/8.d0

!     Tp   = Tend*5d-1
!     sigmaT = Tend*2.5d-1

    tolEnvZ = 1.0d-1
    zf = spanZ/2.d0
    sigmaZ = dsqrt(-(zf**2.d0)/(2.d0*dlog(tolEnvZ)))
    normMode = 1.d0/(nmodes*(bmodes*2.d0 + 1.d0))

    do ix=1,nx
      envX = dexp(-(x(ix)-xC)**(2.d0)/(2.d0*sigmaX**(2.d0)))

      do iz=1,nz
        do jmode=-bmode,bmode
          do imode=1,nmode

            funcZ = exp(iImag*(jmode*beta0*z(iz)))
            funcT = exp(iImag*(imode*omega0*(t)))
            phase = exp(iImag*phi(imode,jmode+bmode+1))

            u(ix,:,iz) = u(ix,:,iz) + real(funcT*funcZ*phase)*envX

          enddo !imode=0,nmode
        enddo !jmode=-bmode,bmode
      enddo !iz=1,nz
    enddo !ix=1,nx

    u = u*amplitude*normMode

    end subroutine
