#!/bin/bash

# Function to download a file from Google Drive (handles large files)
download_from_gdrive () {
    FILE_ID=$1
    FILE_NAME=$2
    EXPECTED_MD5=$3

    echo "Downloading ${FILE_NAME}..."

    # Get confirmation token
    CONFIRM=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies \
        --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- \
        | sed -n 's/.*confirm=\(.*\)&amp;.*/\1/p')

    # Download the actual file
    wget --load-cookies cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
        --no-check-certificate -O ${FILE_NAME}

    rm -f cookies.txt

    # Verify MD5 checksum
    echo "Verifying MD5..."
    ACTUAL_MD5=$(md5sum ${FILE_NAME} | awk '{print $1}')
    if [ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]; then
        echo "✅ ${FILE_NAME} downloaded and verified."
    else
        echo "❌ MD5 mismatch for ${FILE_NAME}!"
        echo "Expected: ${EXPECTED_MD5}"
        echo "Actual:   ${ACTUAL_MD5}"
    fi
}



# File 1
download_from_gdrive "1G2YZo1HKkvf5sMvaaRQeZp_YA9GDlvL-" "models/search_code_18590.pt" "55dc7fe4bee4a93d25a1dd329dfdd159"

# File 2
download_from_gdrive "1kH0yTyiXUNqdYXpnSXI2p5-vFwDYoCzl" "models/model.pt" "e21d663b33d29a8ca4e28d60a14a3d66"

# File 3
download_from_gdrive "1qwV99IfYvLzpjsHgfrlciqDp8yQ5uLXs" "models/model.ts" "7c08cb505b719914015d9fae746bea7c"
