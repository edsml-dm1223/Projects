class PostcodeProcessing:
    def __init__(self, postcode= None, sector= None, district= None):
        self.postcode = postcode
        self.sector = sector
        self.district = district
        pass

    #Function to convert the postcode to a sector
    def postcode_to_sector(self):
        ''' A function to convert a postcode to a sector by extracting the outward code and the first digit of the inward code.
            Arguments:
            postcode (str): A UK postcode
        
            Returns:
            str: The postcode sector
        
            Example:
            postcode_to_sector('SW1A 1AA')
            'SW1A 1'
        '''
        # Cleaning the postcode by removing leading/trailing white spaces and converting to uppercase
        postcode = re.sub(r'\s+', ' ', self.postcode.strip()).upper()
    
        # Validating the postcode format using a regex
        match = re.match(
            r'^([A-Z]{1,2}\d[A-Z\d]?) ?(\d[A-Z]{2})$', self.postcode
        )
    
        # Returning None if the postcode is invalid to make the function robust
        if not match:
            return None
    
        # Extracting the outward code and the first digit of the inward code
        outward, inward = match.groups()
        return f"{outward} {inward[0]}"

    #Function to convert the postcode to a district
    def postcode_to_district(self, postcode):
        ''' A function to convert a postcode to a district by extracting the outward code.
            Arguments:
            postcode (str): A UK postcode
        
            Returns:
            str: The postcode district
        
            Example:
            postcode_to_district('SW1A 1AA')
            'SW1A'
        '''
        # Cleaning the postcode by removing leading/trailing white spaces and converting to uppercase
        postcode = re.sub(r'\s+', ' ', self.postcode.strip()).upper()
    
        # Validating the postcode format
        match = re.match(
            r'^([A-Z]{1,2}\d[A-Z\d]?) ?(\d[A-Z]{2})$', self.postcode
        )
    
        # Returning None if the postcode is invalid to make the function robust
        if not match:
            return None
    
        # Extracting the outward code
        outward, inward = match.groups()
        return f"{outward}"

    def postcode_to_district(self):
        ''' A function to convert a postcode to a district by extracting the outward code.
            Arguments:
            postcode (str): A UK postcode
        
            Returns:
            str: The postcode district
        
            Example:
            postcode_to_district('SW1A 1AA')
            'SW1A'
        '''
        postcode = re.sub(r'\s+', ' ', self.postcode.strip()).upper()
    
        # Validating the postcode format
        match = re.match(
            r'^([A-Z]{1,2}\d[A-Z\d]?) ?(\d[A-Z]{2})$', self.postcode
        )
    
        # Returning None if the postcode is invalid to make the function robust
        if not match:
            return None
    
        # Extracting the outward code
        outward, inward = match.groups()
        return f"{outward}"

    # Cleaning the sector data
    def normalise_sector(self):
        ''' A function to normalise the postcode sector by removing leading/trailing white spaces and converting to uppercase.
            Arguments:
            sector (str): A UK postcode sector
        
            Returns:
            str: The normalised postcode sector
        
            Example:
            normalise_sector('SW1A   1')
            'SW1A 1'
        '''
        sector = re.sub(r'\s+', ' ', self.sector.strip()).upper()
    
        # Validating the postcode format using a regex
        match = re.match(r'^([A-Z]{1,2}[0-9][A-Z0-9]?)\s?([0-9])$', sector)
    
        # Returning None if the postcode is invalid to make the function robust
        if not match:
            return None
    
        # Extracting the outward code and the first digit of the inward code
        outward, inward = match.groups()
        return f"{outward} {inward[0]}"
    
    # Cleaning the district data
    def normalise_district(self):
        ''' A function to normalise the postcode district by removing leading/trailing white spaces and converting to uppercase.
            Arguments:
            district (str): A UK postcode district
        
            Returns:
            str: The normalised postcode district
        
            Example:
            normalise_district('SW1A   ')
            'SW1A'
        '''
        if not isinstance(district, str):
            raise ValueError("Input must be a string")
    
        # Clean and uppercase the input
        district = re.sub(r'\s+', ' ', self.district.strip()).upper()
    
        # Validate the district format
        match = re.match(r'^([A-Z]{1,2}\d{1,2})$', self.district)
    
        if not match:
            return None  # Return None for invalid districts
    
        # Return the normalized district
        return match.group(1)
    
