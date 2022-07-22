CREATE DATABASE media;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

CREATE SCHEMA IF NOT EXISTS `default_schema` ;
USE `default_schema` ;

-- -----------------------------------------------------
-- Table `default_schema`.`author`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`author` (
  `author_id` INT NOT NULL ,
  `f_name` TINYTEXT NOT NULL ,
  `l_name` TINYTEXT NOT NULL ,
  `bio` MEDIUMTEXT NULL ,
  PRIMARY KEY (`author_id`) );


-- -----------------------------------------------------
-- Table `default_schema`.`country`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`country` (
  `country_name` TINYTEXT NOT NULL ,
  `country_abrv` VARCHAR(2) NOT NULL ,
  PRIMARY KEY (`country_abrv`) );


-- -----------------------------------------------------
-- Table `default_schema`.`state`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`state` (
  `state_abrv` VARCHAR(2) NOT NULL ,
  `state_name` TINYTEXT NOT NULL ,
  PRIMARY KEY (`state_abrv`) );


-- -----------------------------------------------------
-- Table `default_schema`.`publisher`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`publisher` (
  `publisher_name` VARCHAR(100) NOT NULL ,
  `country_country_abrv` VARCHAR(2) NOT NULL ,
  `city` TINYTEXT NOT NULL ,
  `street` TINYTEXT NOT NULL ,
  `zip` TINYTEXT NOT NULL ,
  `state_state_abrv` VARCHAR(2) NOT NULL ,
  PRIMARY KEY (`publisher_name`) ,
  INDEX `Publisher_country` (`country_country_abrv` ASC) ,
  INDEX `fk_publisher_state1_idx` (`state_state_abrv` ASC) ,
  CONSTRAINT `Publisher_country`
    FOREIGN KEY (`country_country_abrv` )
    REFERENCES `default_schema`.`country` (`country_abrv` ),
  CONSTRAINT `fk_publisher_state1`
    FOREIGN KEY (`state_state_abrv` )
    REFERENCES `default_schema`.`state` (`state_abrv` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);


-- -----------------------------------------------------
-- Table `default_schema`.`book_info`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`book_info` (
  `ISBN` VARCHAR(10) NOT NULL ,
  `yr_publication` YEAR NOT NULL ,
  `publisher_publisher_name` VARCHAR(100) NOT NULL ,
  `author_author_id` INT NOT NULL ,
  `title` TEXT NOT NULL ,
  `desc` MEDIUMTEXT NULL ,
  `number_of_pages` INT NOT NULL ,
  `cover_url` TEXT NULL,
  PRIMARY KEY (`ISBN`) ,
  INDEX `Media_Publisher` (`publisher_publisher_name` ASC) ,
  INDEX `book_info_Author` (`author_author_id` ASC) ,
  CONSTRAINT `Media_Publisher`
    FOREIGN KEY (`publisher_publisher_name` )
    REFERENCES `default_schema`.`publisher` (`publisher_name` ),
  CONSTRAINT `book_info_Author`
    FOREIGN KEY (`author_author_id` )
    REFERENCES `default_schema`.`author` (`author_id` ));


-- -----------------------------------------------------
-- Table `default_schema`.`book_Inventory`
-- -----------------------------------------------------
CREATE  TABLE IF NOT EXISTS `default_schema`.`book_Inventory` (
  `Inventory_id` VARCHAR(20) NOT NULL ,
  `book_info_ISBN` VARCHAR(10) NOT NULL ,
  `Checked_out` TINYINT(1) NOT NULL ,
  `Checked_out_by` TINYTEXT NULL DEFAULT NULL ,
  `checked_out_date` DATE NULL DEFAULT NULL ,
  PRIMARY KEY (`Inventory_id`) ,
  INDEX `Inventory_Media` (`book_info_ISBN` ASC) ,
  CONSTRAINT `Inventory_Media`
    FOREIGN KEY (`book_info_ISBN` )
    REFERENCES `default_schema`.`book_info` (`ISBN` ));

USE `default_schema` ;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

-- -----------------------------------------------------

-- Data for table `default_schema`.`author`

-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`author` (`author_id`, `f_name`, `l_name`, `bio`) VALUES (1, 'Lewis', 'Carroll', 'Lewis Carroll (1832-1898), the pen name of Oxford mathematician, logician, photographer and author Charles Lutwidge Dodgson, is famous the world over for his fantastic classics \"Alice\'s Adventures in Wonderland,\" \"Through the Looking Glass,\" \"The Hunting of the Snark,\" \"Jabberwocky,\" and \"Sylvie and Bruno.\"');
INSERT INTO `default_schema`.`author` (`author_id`, `f_name`, `l_name`, `bio`) VALUES (2, 'J.D.', 'Sallinger', 'Born in New York in 1919, Jerome David Salinger dropped out of several schools before enrolling in a writing class at Columbia University, publishing his first piece (\"The Young Folks\") in Story magazine. Soon after, the New Yorker picked up the heralded \"A Perfect Day for Bananafish,\" and more pieces followed, including \"Slight Rebellion off Madison\" in 1941, an early Holden Caulfield story. Following a stint in Europe for World War II, Salinger returned to New York and began work on his signature novel, 1951\'s \"The Catcher in the Rye,\" an immediate bestseller for its iconoclastic hero and forthright use of profanity. Following this success, Salinger retreated to his Cornish, New Hampshire, home where he grew increasingly private, eventually erecting a wall around his property and publishing just three more books: \"Nine Stories,\" \"Franny and Zooey,\" \"Raise High the Roof Beam, and Carpenters and Seymour: An Introduction.\" Salinger was married twice and had two children. He died of natural causes on January 27, 2010, in New Hampshire at the age of 91.');
INSERT INTO `default_schema`.`author` (`author_id`, `f_name`, `l_name`, `bio`) VALUES (3, 'F. Scott', 'Fitzgerald', 'F. Scott Fitzgerald was one of the major American writers of the twentieth century -- a figure whose life and works embodied powerful myths about our national dreams and aspirations. Fitzgerald was talented and perceptive, gifted with a lyrical style and a pitch-perfect ear for language. He lived his life as a romantic, equally capable of great dedication to his craft and reckless squandering of his artistic capital. He left us one sure masterpiece, The Great Gatsby; a near-masterpiece, Tender Is the Night; and a gathering of stories and essays that together capture the essence of the American experience. His writings are insightful and stylistically brilliant; today he is admired both as a social chronicler and a remarkably gifted artist.');
INSERT INTO `default_schema`.`author` (`author_id`, `f_name`, `l_name`, `bio`) VALUES (4, 'Harper', 'Lee', 'Harper Lee was born in 1926 in Monroeville, Alabama. She is the author of the acclaimed novels To Kill a Mockingbird and Go Set a Watchman, and was awarded the Pulitzer Prize, the Presidential Medal of Freedom, and numerous other literary awards and honors. She died on February 19, 2016.');


COMMIT;


-- -----------------------------------------------------

-- Data for table `default_schema`.`country`

-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Afghanistan', 'AF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ã…land Islands', 'AX');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Albania', 'AL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Algeria', 'DZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('American Samoa', 'AS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Andorra', 'AD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Angola', 'AO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Anguilla', 'AI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Antarctica', 'AQ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Antigua and Barbuda', 'AG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Argentina', 'AR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Armenia', 'AM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Aruba', 'AW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Australia', 'AU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Austria', 'AT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Azerbaijan', 'AZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bahamas', 'BS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bahrain', 'BH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bangladesh', 'BD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Barbados', 'BB');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Belarus', 'BY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Belgium', 'BE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Belize', 'BZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Benin', 'BJ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bermuda', 'BM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bhutan', 'BT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bolivia, Plurinational State of', 'BO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bonaire, Sint Eustatius and Saba', 'BQ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bosnia and Herzegovina', 'BA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Botswana', 'BW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bouvet Island', 'BV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Brazil', 'BR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('British Indian Ocean Territory', 'IO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Brunei Darussalam', 'BN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Bulgaria', 'BG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Burkina Faso', 'BF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Burundi', 'BI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cambodia', 'KH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cameroon', 'CM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Canada', 'CA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cape Verde', 'CV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cayman Islands', 'KY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Central African Republic', 'CF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Chad', 'TD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Chile', 'CL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('China', 'CN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Christmas Island', 'CX');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cocos (Keeling) Islands', 'CC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Colombia', 'CO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Comoros', 'KM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Congo', 'CG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Congo, the Democratic Republic of the', 'CD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cook Islands', 'CK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Costa Rica', 'CR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('CÃ´te d\'Ivoire', 'CI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Croatia', 'HR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cuba', 'CU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('CuraÃ§ao', 'CW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Cyprus', 'CY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Czech Republic', 'CZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Denmark', 'DK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Djibouti', 'DJ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Dominica', 'DM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Dominican Republic', 'DO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ecuador', 'EC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Egypt', 'EG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('El Salvador', 'SV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Equatorial Guinea', 'GQ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Eritrea', 'ER');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Estonia', 'EE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ethiopia', 'ET');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Falkland Islands (Malvinas)', 'FK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Faroe Islands', 'FO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Fiji', 'FJ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Finland', 'FI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('France', 'FR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('French Guiana', 'GF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('French Polynesia', 'PF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('French Southern Territories', 'TF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Gabon', 'GA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Gambia', 'GM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Georgia', 'GE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Germany', 'DE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ghana', 'GH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Gibraltar', 'GI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Greece', 'GR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Greenland', 'GL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Grenada', 'GD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guadeloupe', 'GP');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guam', 'GU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guatemala', 'GT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guernsey', 'GG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guinea', 'GN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guinea-Bissau', 'GW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Guyana', 'GY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Haiti', 'HT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Heard Island and McDonald Islands', 'HM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Holy See (Vatican City State)', 'VA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Honduras', 'HN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Hong Kong', 'HK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Hungary', 'HU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Iceland', 'IS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('India', 'IN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Indonesia', 'ID');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Iran, Islamic Republic of', 'IR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Iraq', 'IQ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ireland', 'IE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Isle of Man', 'IM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Israel', 'IL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Italy', 'IT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Jamaica', 'JM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Japan', 'JP');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Jersey', 'JE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Jordan', 'JO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Kazakhstan', 'KZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Kenya', 'KE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Kiribati', 'KI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Korea, Democratic People\'s Republic of', 'KP');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Korea, Republic of', 'KR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Kuwait', 'KW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Kyrgyzstan', 'KG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Lao People\'s Democratic Republic', 'LA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Latvia', 'LV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Lebanon', 'LB');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Lesotho', 'LS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Liberia', 'LR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Libya', 'LY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Liechtenstein', 'LI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Lithuania', 'LT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Luxembourg', 'LU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Macao', 'MO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Macedonia, the Former Yugoslav Republic of', 'MK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Madagascar', 'MG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Malawi', 'MW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Malaysia', 'MY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Maldives', 'MV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mali', 'ML');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Malta', 'MT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Marshall Islands', 'MH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Martinique', 'MQ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mauritania', 'MR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mauritius', 'MU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mayotte', 'YT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mexico', 'MX');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Micronesia, Federated States of', 'FM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Moldova, Republic of', 'MD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Monaco', 'MC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mongolia', 'MN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Montenegro', 'ME');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Montserrat', 'MS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Morocco', 'MA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Mozambique', 'MZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Myanmar', 'MM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Namibia', 'NA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Nauru', 'NR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Nepal', 'NP');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Netherlands', 'NL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('New Caledonia', 'NC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('New Zealand', 'NZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Nicaragua', 'NI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Niger', 'NE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Nigeria', 'NG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Niue', 'NU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Norfolk Island', 'NF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Northern Mariana Islands', 'MP');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Norway', 'NO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Oman', 'OM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Pakistan', 'PK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Palau', 'PW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Palestine, State of', 'PS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Panama', 'PA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Papua New Guinea', 'PG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Paraguay', 'PY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Peru', 'PE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Philippines', 'PH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Pitcairn', 'PN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Poland', 'PL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Portugal', 'PT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Puerto Rico', 'PR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Qatar', 'QA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('RÃ©union', 'RE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Romania', 'RO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Russian Federation', 'RU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Rwanda', 'RW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint BarthÃ©lemy', 'BL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Helena, Ascension and Tristan da Cunha', 'SH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Kitts and Nevis', 'KN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Lucia', 'LC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Martin (French part)', 'MF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Pierre and Miquelon', 'PM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saint Vincent and the Grenadines', 'VC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Samoa', 'WS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('San Marino', 'SM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sao Tome and Principe', 'ST');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Saudi Arabia', 'SA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Senegal', 'SN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Serbia', 'RS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Seychelles', 'SC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sierra Leone', 'SL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Singapore', 'SG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sint Maarten (Dutch part)', 'SX');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Slovakia', 'SK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Slovenia', 'SI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Solomon Islands', 'SB');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Somalia', 'SO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('South Africa', 'ZA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('South Georgia and the South Sandwich Islands', 'GS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('South Sudan', 'SS');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Spain', 'ES');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sri Lanka', 'LK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sudan', 'SD');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Suriname', 'SR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Svalbard and Jan Mayen', 'SJ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Swaziland', 'SZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Sweden', 'SE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Switzerland', 'CH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Syrian Arab Republic', 'SY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Taiwan, Province of China', 'TW');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tajikistan', 'TJ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tanzania, United Republic of', 'TZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Thailand', 'TH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Timor-Leste', 'TL');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Togo', 'TG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tokelau', 'TK');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tonga', 'TO');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Trinidad and Tobago', 'TT');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tunisia', 'TN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Turkey', 'TR');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Turkmenistan', 'TM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Turks and Caicos Islands', 'TC');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Tuvalu', 'TV');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Uganda', 'UG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Ukraine', 'UA');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('United Arab Emirates', 'AE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('United Kingdom', 'GB');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('United States', 'US');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('United States Minor Outlying Islands', 'UM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Uruguay', 'UY');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Uzbekistan', 'UZ');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Vanuatu', 'VU');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Venezuela, Bolivarian Republic of', 'VE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Viet Nam', 'VN');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Virgin Islands, British', 'VG');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Virgin Islands, U.S.', 'VI');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Wallis and Futuna', 'WF');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Western Sahara', 'EH');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Yemen', 'YE');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Zambia', 'ZM');
INSERT INTO `default_schema`.`country` (`country_name`, `country_abrv`) VALUES ('Zimbabwe', 'ZW');

COMMIT;


-- -----------------------------------------------------
-- Data for table `default_schema`.`state`
-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('AL', 'Alabama');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('AK', 'Alaska');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('AZ', 'Arizona');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('AR', 'Arkansas');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('CA', 'California');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('CO', 'Colorado');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('CT', 'Connecticut');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('DE', 'Delaware');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('DC', 'District of Columbia');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('FL', 'Florida');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('GA', 'Georgia');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('HI', 'Hawaii');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('ID', 'Idaho');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('IL', 'Illinois');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('IN', 'Indiana');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('IA', 'Iowa');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('KS', 'Kansas');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('KY', 'Kentucky');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('LA', 'Louisiana');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('ME', 'Maine');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MT', 'Montana');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NE', 'Nebraska');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NV', 'Nevada');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NH', 'New Hampshire');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NJ', 'New Jersey');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NM', 'New Mexico');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NY', 'New York');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NC', 'North Carolina');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('ND', 'North Dakota');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('OH', 'Ohio');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('OK', 'Oklahoma');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('OR', 'Oregon');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MD', 'Maryland');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MA', 'Massachusetts');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MI', 'Michigan');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MN', 'Minnesota');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MS', 'Mississippi');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('MO', 'Missouri');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('PA', 'Pennsylvania');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('RI', 'Rhode Island');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('SC', 'South Carolina');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('SD', 'South Dakota');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('TN', 'Tennessee');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('TX', 'Texas');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('UT', 'Utah');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('VT', 'Vermont');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('VA', 'Virginia');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('WA', 'Washington');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('WV', 'West Virginia');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('WI', 'Wisconsin');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('WY', 'Wyoming');
INSERT INTO `default_schema`.`state` (`state_abrv`, `state_name`) VALUES ('NA', 'Not Applicable');

COMMIT;

-- -----------------------------------------------------
-- Data for table `default_schema`.`publisher`
-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`publisher` (`publisher_name`, `country_country_abrv`, `city`, `street`, `zip`, `state_state_abrv`) VALUES ('CreateSpace', 'US', 'North Charleston', '4900 Lacross Rd', '29406-6558', 'SC');
INSERT INTO `default_schema`.`publisher` (`publisher_name`, `country_country_abrv`, `city`, `street`, `zip`, `state_state_abrv`) VALUES ('Little, Brown and Company', 'US', 'Boston', '53 State St', '02109', 'MA');
INSERT INTO `default_schema`.`publisher` (`publisher_name`, `country_country_abrv`, `city`, `street`, `zip`, `state_state_abrv`) VALUES ('Simon and Schuster', 'US', 'New York', '1230 Avenue of the Americas, 10th F', '10020', 'NY');
INSERT INTO `default_schema`.`publisher` (`publisher_name`, `country_country_abrv`, `city`, `street`, `zip`, `state_state_abrv`) VALUES ('HarperCollins', 'US', 'New York', '331 Pearl Street', '10038', 'NY');

COMMIT;


-- -----------------------------------------------------
-- Data for table `default_schema`.`book_info`
-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('1503222683', '2019', 'CreateSpace', 1, 'Alice\'s Adventures in Wonderland', 'Alice\'s Adventures in Wonderland is an 1865 novel written by English author Charles Lutwidge Dodgson under the pseudonym Lewis Carroll. It tells of a girl named Alice falling through a rabbit hole into a fantasy world populated by peculiar, anthropomorphic creatures. The tale plays with logic, giving the story lasting popularity with adults as well as with children. It is considered to be one of the best examples of the literary nonsense genre. Its narrative course and structure, characters and imagery have been enormously influential in both popular culture and literature, especially in the fantasy genre.', 70, 'https://images-na.ssl-images-amazon.com/images/I/71su1O1yFJL.jpg');
INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('7543321726', '1991', 'Little, Brown and Company', 2, 'The Catcher in the Rye', 'Anyone who has read J.D. Salinger\'s New Yorker stories, particularly A Perfect Day for Bananafish, Uncle Wiggily in Connecticut, The Laughing Man, and For Esme--With Love and Squalor, will not be surprised by the fact that his first novel is full of children. The hero-narrator of THE CATCHER IN THE RYE is an ancient child of sixteen, a native New Yorker named Holden Caulfield. Through circumstances that tend to preclude adult, secondhand description, he leaves his prep school in Pennsylvania and goes underground in New York City for three days. ', 240, 'https://images-na.ssl-images-amazon.com/images/I/81OthjkJBuL.jpg');
INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('0743273567', '2003', 'Simon and Schuster', 3, 'The Great Gatsby', 'A true classic of twentieth-century literature, this edition has been updated by Fitzgerald scholar James L.W. West III to include the author’s final revisions and features a note on the composition and text, a personal foreword by Fitzgerald’s granddaughter, Eleanor Lanahan—and a new introduction by two-time National Book Award winner Jesmyn Ward.', 76, 'https://images-na.ssl-images-amazon.com/images/I/41iers%2BHLSL._SY346_.jpg');
INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('0451530438', '2007', 'CreateSpace', 3, 'The Beautiful and Damned', 'Set in an era of intoxicating excitement and ruinous excess, changing manners and challenged morals, F. Scott Fitzgerald’s second novel chronicles the lives of Harvard-educated Anthony Patch and his beautiful, willful wife, Gloria. This bitingly ironic story eerily foretells the fate of the author and his own wife, Zelda—from its giddy romantic beginnings to its alcohol-fueled demise. A portrait of greed, ambition, and squandered talent, The Beautiful and Damned depicts an America embarked on the greatest spree in its history, a world Fitzgerald saw “with clearer eyes than any of his contemporaries.”* By turns hilarious, heartbreaking, and chillingly prophetic, it remains one of his best-known works, which Gertrude Stein correctly predicted “will be read when many of his well-known contemporaries are forgotten.”', 400, 'https://images-na.ssl-images-amazon.com/images/I/817lz-3nQUL.jpg');
INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('9780062420', '1960', 'HarperCollins', 4, 'To Kill a Mockingbiird', 'One of the most cherished stories of all time, To Kill a Mockingbird has been translated into more than forty languages, sold more than forty million copies worldwide, served as the basis for an enormously popular motion picture, and was voted one of the best novels of the twentieth century by librarians across the country. A gripping, heart-wrenching, and wholly remarkable tale of coming-of-age in a South poisoned by virulent prejudice, it views a world of great beauty and savage inequities through the eyes of a young girl, as her father—a crusading local lawyer—risks everything to defend a black man unjustly accused of a terrible crime.', 336, 'https://images-na.ssl-images-amazon.com/images/I/51Dv42myl0L.jpg');
INSERT INTO `default_schema`.`book_info` (`ISBN`, `yr_publication`, `publisher_publisher_name`, `author_author_id`, `title`, `desc`, `number_of_pages`, `cover_url`) VALUES ('9780062409', '2015', 'HarperCollins', 4, 'Go Set a Watchman:A Novel', 'Twenty-six-year-old Jean Louise Finch—“Scout”—returns home to Maycomb, Alabama from New York City to visit her aging father, Atticus. Set against the backdrop of the civil rights tensions and political turmoil that were transforming the South, Jean Louise’s homecoming turns bittersweet when she learns disturbing truths about her close-knit family, the town, and the people dearest to her. Memories from her childhood flood back, and her values and assumptions are thrown into doubt. Featuring many of the iconic characters from To Kill a Mockingbird, Go Set a Watchman perfectly captures a young woman, and a world, in painful yet necessary transition out of the illusions of the past—a journey that can only be guided by one’s own conscience.', 288, 'https://images-na.ssl-images-amazon.com/images/I/51sbtF6KaPL.jpg');

COMMIT;


-- -----------------------------------------------------
-- Data for table `default_schema`.`book_Inventory`
-- -----------------------------------------------------

START TRANSACTION;

USE `default_schema`;

INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('1503222683_1', '1503222683', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('1503222683_2', '1503222683', 1, 'alabamajeff', '2019-04-30');
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('7543321726_1', '7543321726', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('7543321726_2', '7543321726', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('7543321726_3', '7543321726', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0743273567_1', '0743273567', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0743273567_2', '0743273567', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0743273567_3', '0743273567', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0743273567_4', '0743273567', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0743273567_5', '0743273567', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0451530438_1', '0451530438', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('0451530438_2', '0451530438', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062420_1', '9780062420', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062420_2', '9780062420', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062420_3', '9780062420', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062409_1', '9780062409', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062409_2', '9780062409', 0, NULL, NULL);
INSERT INTO `default_schema`.`book_Inventory` (`Inventory_id`, `book_info_ISBN`, `Checked_out`, `Checked_out_by`, `checked_out_date`) VALUES ('9780062409_3', '9780062409', 0, NULL, NULL);

COMMIT;