#!/bin/bash
set -e

green="\033[00;32m"
red="\033[0;31m"
txtrst="\033[00;0m"

#Check OS Version
os_codename=`cat /etc/os-release | grep -e VERSION_CODENAME | awk -F '=' '{print $2}' | xargs`
if [ "$os_codename" != "bookworm" ] && [ "$os_codename" != "bullseye" ]; then
        echo -e "${red}The current OS isn't compatible with VitalPBX...${txtrst}"
        exit ;
fi

pbx_version="v4"
if [ "$os_codename" == "bookworm" ]; then
	echo -e "${green}Installing on Debian 12...${txtrst}"
	pbx_version="v4.5"
else
	echo -e "${green}Installing on Debian 11...${txtrst}"
fi

#Install sudo
apt install sudo -y

#Disable UFW
if [ -x "$(command -v ufw)" ]; then
	echo -e "${green}Disabling UFW...${txtrst}"
	sudo ufw disable
fi

#Initial update
echo -e "${green}Updating the System...${txtrst}"
sudo apt -y update
sudo DEBIAN_FRONTEND=noninteractive apt -y upgrade
sudo apt -y clean

#Install curl gpg
sudo DEBIAN_FRONTEND=noninteractive apt install curl gpg -y

#Installin VitalPBX Repo
echo -e "${green}Installing VitalPBX Repo...${txtrst}"
curl -fsSL https://repo.vitalpbx.com/vitalpbx/${pbx_version}/apt/setup_repo | bash -

#Update System
echo -e "${green}Update after setup VPBX Repo...${txtrst}"
sudo apt -y update
sudo DEBIAN_FRONTEND=noninteractive apt -y upgrade
sudo apt -y clean

echo -e "${green}Installing dependencies...${txtrst}"

#System Packages
PACKAGES="acl cron lame sox ffmpeg aptitude postfix nmap net-tools systemd-timesyncd"

if [ "$os_codename" == "bookworm" ]; then
	PACKAGES="$PACKAGES nginx php-fpm ssl-cert apache2-utils"
else
	PACKAGES="$PACKAGES apache2 php8.1-fpm"
fi

#Web Pabackages
PACKAGES="$PACKAGES php-common php-xml php-intl php-pear php-bcmath php-cli php-mysql php-zip"
PACKAGES="$PACKAGES php-gd php-mbstring php-json php-imagick php-curl php-ioncube-loader openvpn unzip"

# Database
PACKAGES="$PACKAGES mariadb-server unixodbc odbcinst unixodbc-dev"

sudo DEBIAN_FRONTEND=noninteractive apt install -y $PACKAGES

#Web Server Configuration
if [ "$os_codename" == "bookworm" ]; then
	#Configuring NGINX
	echo -e "${green}Configuring NGINX...${txtrst}"
	systemctl enable nginx.service
	systemctl restart nginx.service
else
	#Configuring Apache
	echo -e "${green}Configuring Apache...${txtrst}"
	sudo a2dismod -q php8.1 || :
	sudo a2enconf -q php8.1-fpm || :
	sudo a2enmod -q proxy_fcgi || :
	sudo a2enmod -q proxy_http || :
	sudo a2enmod -q ssl || :
	sudo a2enmod -q headers || :
	sudo a2enmod -q rewrite || :

	echo -e "${green}Enabling HTTP/2 support for Apache...${txtrst}"
	a2dismod -q mpm_prefork || :
	a2enmod -q mpm_event || :
	a2enmod -q ssl || :
	a2enmod -q http2 || :

	systemctl enable apache2.service
	systemctl restart apache2.service
fi

#Starting & Enabling Services
echo -e "${green}Starting & Enabling Services...${txtrst}"
systemctl enable mariadb.service
systemctl restart mariadb.service

#Installing VitalPBX pacakges
echo -e "${green}Installing VitalPBX packages and Firewall...${txtrst}"

#Security Packages
PACKAGES="asterisk-pbx asterisk-pbx-modules asterisk-pbx-g729 sngrep vitalpbx-asterisk-config vitalpbx-fail2ban-config vitalpbx-api"
PACKAGES="$PACKAGES firewalld fail2ban iptables iptables-persistent"
PACKAGES="$PACKAGES vitalpbx-i18n vitalpbx-helper vitalpbx-assets vitalpbx-scripts vitalpbx-monitor vitalpbx"

sudo DEBIAN_FRONTEND=noninteractive apt install -y $PACKAGES

#Configuring Firewall
echo -e "${green}Configuring Firewall...${txtrst}"
sed -i 's/^FirewallBackend=.*/FirewallBackend=iptables/g' /etc/firewalld/firewalld.conf
update-alternatives --set iptables /usr/sbin/iptables-legacy
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

systemctl enable firewalld.service

systemctl enable fail2ban.service

#Disable IPTables to Avoid Conflicts with FirewallD
systemctl disable iptables 2>/dev/null

#Fix DNSMASQ
if [ "$os_codename" == "bookworm" ]; then
	sed -i 's/^#bind-interfaces/bind-interfaces/g' /etc/dnsmasq.conf
fi

#Set Permissions
chown -R asterisk:asterisk /etc/asterisk /var/lib/asterisk /var/log/asterisk /var/spool/asterisk /usr/lib/asterisk
chown -R www-data:root /etc/asterisk/vitalpbx

#Running Initial setup
echo -e "${green}Running the initial setup...${txtrst}"
systemctl enable vpbx-setup.service

#Configure Packages
dpkg --configure -a

#Rebooting the System
echo -e "${green}Rebooting the System...${txtrst}"
sudo reboot
