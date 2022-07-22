<template>
  <div class="content">
    <div v-if="$keycloak.hasRealmRole('librarian')">
      <div class="md-layout">
        <md-card class="md-card-plain">
          <md-card-header data-background-color="blue">
            <h4 class="title">Add Book</h4>
          </md-card-header>
          <md-card-content>
            <h5> Which book would you like to add? </h5>
            <md-field>
              <label>Book</label>
              <md-select
                v-model="BookSelected"
                name="BookSelected"
                id="BookSelected"
                placeholder="Select Book"
              >
                <md-option value="new"> New Book </md-option>
                <md-option
                  v-bind:key="Book"
                  v-for="Book in BookData" 
                  :value="Book.ISBN + '_' + (Book.Total+1)"
                >
                  {{ Book.Title }}
                </md-option>
              </md-select>
            </md-field>
            <div v-if="BookSelected == 'new'">
              <h5> Please enter the information for the new book! </h5>
                <div class="md-layout md-gutter">
                  <div class="md-layout-item md-small-size-100">
                    <md-field>
                      <label> ISBN </label>
                      <md-input name="ISBN" id="ISBN" v-model="bookForm.ISBN" />
                    </md-field>
                  </div>
                </div>
                <div class="md-layout md-gutter">
                  <div class="md-layout-item md-small-size-100">
                    <md-field>
                      <label> Title </label>
                      <md-input name="Title" id="Title" v-model="bookForm.Title" />
                    </md-field>
                  </div>
                </div>
                <div class="md-layout md-gutter">
                  <div class="md-layout-item md-small-size-100">
                    <md-field>
                      <label> Cover Url </label>
                      <md-input name="Cover" id="Cover" v-model="bookForm.Cover" />
                    </md-field>
                  </div>
                </div>
                <div class="md-layout md-gutter">
                  <div class="md-layout-item md-small-size-100">
                    <md-field>
                      <label> Description </label>
                      <md-textarea name="Desc" id="Desc" v-model="bookForm.Desc" />
                    </md-field>
                  </div>
                </div>
              <div class="md-layout md-gutter">
                <div class="md-layout-item md-small-size-25">
                  <md-field>
                    <label> # of Pages </label>
                    <md-input name="Page" id="Page" v-model="bookForm.pages" />
                  </md-field>
                </div>
                <div class="md-layout-item md-small-size-25">
                  <md-field>
                    <label> Publicaion Year </label>
                    <md-input name="year" id="year" v-model="bookForm.year" />
                  </md-field>
                </div>
                <div class="md-layout-item md-small-size-25">
                  <md-field>
                    <label> Publisher </label>
                    <md-select
                      v-model="bookForm.Publisher"
                      name="PublisherSelected"
                      id="PublisherSelected"
                      placeholder="Select Publisher"
                    >
                      <md-option value="new"> New Publisher </md-option>
                      <md-option
                        v-bind:key="Publisher"
                        v-for="Publisher in PublisherData"
                        :value="Publisher.Publisher_name"
                      >
                        {{ Publisher.Publisher_name }}
                      </md-option>
                    </md-select>
                  </md-field>
                </div>
                <div class="md-layout-item md-small-size-25">
                  <md-field>
                    <label> Author </label>
                    <md-select
                      v-model="bookForm.Author"
                      name="AuthorSelected"
                      id="AuthorSelected"
                      placeholder="Select Author"
                    >
                      <md-option :value="AuthorData.length + 1">
                        New Author
                      </md-option>
                      <md-option
                        v-bind:key="Author"
                        v-for="Author in AuthorData"
                        :value="Author.Author_id"
                      >
                        {{ Author.Name }}
                      </md-option>
                    </md-select>
                  </md-field>
                </div>
              </div>
            </div>
            <div
              v-if="
                bookForm.Author == AuthorData.length + 1 &&
                  BookSelected == 'new'
              "
            >
              <md-divider />
              <h5>Enter in new Author info!</h5>
              <div class="md-layout md-gutter">
                <div class="md-layout-item md-small-size-50">
                  <md-field>
                    <label> First Name </label>
                    <md-input
                      name="fname"
                      id="fname"
                      v-model="AuthorForm.fname"
                    />
                  </md-field>
                </div>
                <div class="md-layout-item md-small-size-50">
                  <md-field>
                    <label> First Name </label>
                    <md-input
                      name="lname"
                      id="lname"
                      v-model="AuthorForm.lname"
                    />
                  </md-field>
                </div>
              </div>
              <div class="md-layout md-gutter">
                <div class="md-layout-item md-small-size-100">
                  <md-field>
                    <label> Biography </label>
                    <md-textarea name="Bio" id="Bio" v-model="AuthorForm.Bio" />
                  </md-field>
                </div>
              </div>
            </div>
            <div v-if="bookForm.Publisher == 'new' && BookSelected == 'new'">
              <md-divider />
              <h5>Enter in infor for the new publisher!</h5>
              <div class="md-layout md-gutter">
                <div class="md-layout-item md-size-small-100">
                  <md-field>
                    <label> Name </label>
                    <md-input
                      name="pname"
                      id="pname"
                      v-model="publisherForm.name"
                    />
                  </md-field>
                </div>
              </div>
              <div class="md-layout md-gutter">
                <div class="md-layout-item md-size-small-35">
                  <md-field>
                    <label> Street </label>
                    <md-input
                      name="street"
                      id="street"
                      v-model="publisherForm.street"
                    />
                  </md-field>
                </div>
                <div class="md-layout-item md-size-small-20">
                  <md-field>
                    <label> City </label>
                    <md-input
                      name="city"
                      id="city"
                      v-model="publisherForm.city"
                    />
                  </md-field>
                </div>
                <div class="md-layout-item md-size-small-10">
                  <md-field>
                    <label> Zip </label>
                    <md-input
                      name="zip_c"
                      id="zip_c"
                      v-model="publisherForm.zip_c"
                    />
                  </md-field>
                </div>
                <div class="md-layout-item md-small-size-20">
                  <md-field>
                    <label> Country </label>
                    <md-select
                      v-model="publisherForm.country"
                      name="countrySelected"
                      id="countrySelected"
                      placeholder="Select Country"
                    >
                      <md-option value="US">United States</md-option>
                      <md-option
                        v-bind:key="Country"
                        v-for="Country in CountryData"
                        :value="Country.Country_Abrv"
                      >
                        {{ Country.Name }}
                      </md-option>
                    </md-select>
                  </md-field>
                </div>
                <div
                  v-if="publisherForm.country == 'US'"
                  class="md-layout-item md-small-size-15"
                >
                  <md-field>
                    <label> State </label>
                    <md-select
                      v-model="publisherForm.state"
                      name="stateSelected"
                      id="stateSelected"
                      placeholder="Select State"
                    >
                      <md-option
                        v-for="State in StateData"
                        :key="State"
                        :value="State.State_Abrv"
                      >
                        {{ State.Name }}
                      </md-option>
                    </md-select>
                  </md-field>
                </div>
              </div>
            </div>
            <div class="md-layout md-gutter">
              <div class="md-layout-item"></div>
              <div class="md-layout-item"></div>
              <div class="md-layout-item"></div>
              <div class="md-layout-item"></div>
              <div class="md-layout-item">
                <md-button>Submit</md-button>
              </div>
            </div>
          </md-card-content>
        </md-card>
      </div>
    </div>
    <div v-else>
      <h1>Access Denied</h1>
    </div>
  </div>
</template>

<script>
import axios from "axios";
export default {
  data: () => {
    return {
      BookData: null,
      AuthorData: null,
      PublisherData: null,
      StateData: null,
      CountryData: null,
      BookSelected: null,
      bookForm: {
        ISBN: null,
        Title: null,
        Desc: null,
        Cover: null,
        Publisher: null,
        Author: null,
        year: null,
        pages: null
      },
      AuthorForm: {
        fname: null,
        lname: null,
        bio: null
      },
      publisherForm: {
        name: null,
        street: null,
        city: null,
        zip_c: null,
        state: null,
        country: null
      }
    };
  },
  mounted() {
    axios
      .get("http://localhost:5000/name/")
      .then(response => (this.BookData = response.data.inventory));
    axios
      .get("http://localhost:5000/get_authors/")
      .then(response => (this.AuthorData = response.data));
    axios
      .get("http://localhost:5000/get_publishers")
      .then(response => (this.PublisherData = response.data));
    axios
      .get("http://localhost:5000/states/")
      .then(response => (this.StateData = response.data));
    axios
      .get("http://localhost:5000/countries/")
      .then(response => (this.CountryData = response.data));
  }
};
</script>
