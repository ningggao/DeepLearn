CREATE DATABASE docker_practice_v3;
use docker_practice_v3;

CREATE TABLE `usertable` (
  `id` int NOT NULL AUTO_INCREMENT,
  `UserID` varchar(20),
  `name` varchar(20),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

insert  into `usertable`(`id`,`UserID`,`name`) values
(1,'123123','Apple'),
(2,'321321','Ning');